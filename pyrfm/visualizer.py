import os

import matplotlib.pyplot as plt
import moderngl
import numpy as np
from PIL import Image

from pyrfm.core import *

if torch.cuda.is_available():
    os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
    os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"


class VisualizerBase:
    def __init__(self):
        pass

    def plot(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def show(self, *args, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses.")


class RFMVisualizer(VisualizerBase):
    def __init__(self, model: RFMBase, resolution=(1920, 1080), component_idx=0):
        super().__init__()
        self.model = model
        self.size = resolution
        bounding_box = model.domain.get_bounding_box()
        self.bbox_min = np.array(bounding_box[::2], dtype='f4')
        self.bbox_max = np.array(bounding_box[1::2], dtype='f4')
        self.n_subdomain = model.submodels.numel()
        self.n_hidden = model.n_hidden
        self.dim = model.dim
        self.component_idx = component_idx
        self.centers = model.centers.cpu().numpy().astype('f4')
        self.radii = model.radii.cpu().numpy().astype('f4')
        self.weights = np.array([submodel.weights.cpu().numpy().astype('f4') for submodel in model.submodels.flat_data],
                                dtype='f4')
        self.biases = np.array([submodel.biases.cpu().numpy().astype('f4') for submodel in model.submodels.flat_data],
                               dtype='f4')
        self.outputW = model.W[:, component_idx].cpu().numpy().astype('f4')
        self.vmin, self.vmax = self.estimate_vmin_vmax()
        self.glsl_sdf = model.domain.glsl_sdf()

    @abstractmethod
    def estimate_vmin_vmax(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class RFMVisualizer2D(RFMVisualizer):
    def __init__(self, model: RFMBase, resolution=(1920, 1080), component_idx=0, pou_type=1, act_type=0):
        super().__init__(model, resolution, component_idx)
        self.pou_type = pou_type  # POU type: 0=PsiA, 1=PsiB, 2=PsiBW
        self.act_type = act_type  # Activation type: 0=tanh, 1=sin, 2=cos, 3=ReLU

    def plot(self):
        self._init_gl()
        self.fbo.clear()
        self.vao.render()
        data = self.fbo.read(components=3)
        img = Image.frombytes('RGB', self.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        np_img = np.asarray(img) / 255.0
        fig, ax = plt.subplots()
        ax.imshow(np_img, extent=[self.bbox_min[0], self.bbox_max[0],
                                  self.bbox_min[1], self.bbox_max[1]], origin='lower')
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Scalar Field Value')
        plt.tight_layout()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('2D SDF Field (Viridis)')
        # ax.grid(True);
        plt.tight_layout()
        return ax

    def estimate_vmin_vmax(self, n_samples=100):
        x_sampled = self.model.domain.in_sample(n_samples, with_boundary=True)
        values_sampled = self.model(x_sampled).cpu().numpy().astype('f4')[:, self.component_idx]
        return values_sampled.min(), values_sampled.max()

    def _init_gl(self):
        vert_src = """#version 330
        in vec2 in_vert;
        out vec2 fragCoord;
        void main(){ fragCoord=in_vert; gl_Position=vec4(in_vert,0.0,1.0);}"""
        frag_src = self._build_fragment_shader()
        self.ctx = moderngl.create_standalone_context()
        self.fbo = self.ctx.simple_framebuffer(self.size, components=4)
        self.fbo.use()
        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)
        # uniforms
        # self.prog['iResolution'].value = self.size
        self.prog['bbox_min'].value = self.bbox_min
        self.prog['bbox_max'].value = self.bbox_max
        # --- upload parameter arrays via textures to avoid uniform‑count limits --- #
        # Texture 0: centers (vec2), shape = (n_sub, 1)
        self.centers_tex = self.ctx.texture((self.n_subdomain, 1), 2, data=self.centers.astype('f4').tobytes(),
                                            dtype='f4')
        self.centers_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.centers_tex.use(location=0)
        self.prog['centers_tex'].value = 0

        # Texture 1: radii (vec2), shape = (n_sub, 1)
        self.radii_tex = self.ctx.texture((self.n_subdomain, 1), 2, data=self.radii.astype('f4').tobytes(), dtype='f4')
        self.radii_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.radii_tex.use(location=1)
        self.prog['radii_tex'].value = 1

        # Texture 2: weights (vec2), layout = (row = subfeature, col = hidden neuron)
        w_tex_data = self.weights.transpose(0, 2, 1).astype('f4')  # shape (n_sub, n_hidden, 2)
        self.weights_tex = self.ctx.texture((self.n_hidden, self.n_subdomain), 2, data=w_tex_data.tobytes(), dtype='f4')
        self.weights_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.weights_tex.use(location=2)
        self.prog['weights_tex'].value = 2

        # Texture 3: biases (float), shape = (n_hidden, n_sub)
        self.biases_tex = self.ctx.texture((self.n_hidden, self.n_subdomain), 1,
                                           data=self.biases.astype('f4').tobytes(),
                                           dtype='f4')
        self.biases_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.biases_tex.use(location=3)
        self.prog['biases_tex'].value = 3

        # Texture 4: output weights (float), shape = (n_hidden, n_sub)
        outw = self.outputW.reshape(self.n_subdomain, self.n_hidden).astype('f4')
        self.outputW_tex = self.ctx.texture((self.n_hidden, self.n_subdomain), 1, data=outw.tobytes(), dtype='f4')
        self.outputW_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.outputW_tex.use(location=4)
        self.prog['outputW_tex'].value = 4

        self.prog['vmin'].value = self.vmin
        self.prog['vmax'].value = self.vmax
        quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], 'f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, quad, 'in_vert')

    def _build_fragment_shader(self):
        return f"""
        #version 330
        
        #define DIM       {self.dim}           // 2=2-D, 3=3-D ……
        #define NSUB      {self.n_subdomain}           // 子域数
        #define NHIDDEN   {self.n_hidden}          // 每子域隐藏元
        #define POU_TYPE  {self.pou_type}           // 0=PsiA, 1=PsiB, 2=PsiBW …
        #define ACT_TYPE  {self.act_type}          // 0=tanh, 1=sin, 2=cos, 3=ReLU
        in  vec2 fragCoord;
        out vec4 fragColor;
        
        uniform sampler2D centers_tex;
        uniform sampler2D radii_tex;
        uniform sampler2D weights_tex;   // RG32F : (j,i) => vec2 w
        uniform sampler2D biases_tex;    // R32F  : (j,i) => bias
        uniform sampler2D outputW_tex;   // R32F  : (j,i) => output weight
        uniform vec2  bbox_min;
        uniform vec2  bbox_max;
        uniform float vmin;
        uniform float vmax;
        
        float psiA(float x){{ return abs(x)<=1.0 ? 1.0 : 0.0; }}

        float psiB(float x){{
            return (x<-1.25)?0.0:
                   (x<-0.75)?0.5*(1.0+sin(2.0*3.14159265*x)):
                   (x<=0.75)?1.0:
                   (x<=1.25)?0.5*(1.0-sin(2.0*3.14159265*x)):
                              0.0;
        }}
        
        float psi(float x){{
        #if   POU_TYPE == 0
            return psiA(x);
        #elif POU_TYPE == 1
            return psiB(x);
        #else                // 例如 PsiBW
            return psiB(x);
        #endif
        }}
        
        float act(float z){{
        #if   ACT_TYPE == 0
            return tanh(z);
        #elif ACT_TYPE == 1
            return sin(z);
        #elif ACT_TYPE == 2
            return cos(z);
        #else                       // ReLU
            return max(z, 0.0);
        #endif
        }}
        
        // ---------- viridis colormap ----------
        vec3 viridis(float x) {{
        x = clamp(x, 0.0, 1.0);
        const vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
        const vec3 c1 = vec3(0.190631, 0.407061, 0.556089);
        const vec3 c2 = vec3(0.20803, 0.718701, 0.472873);
        const vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
        return (x < 0.33) ? mix(c0, c1, x / 0.33)
             : (x < 0.66) ? mix(c1, c2, (x - 0.33) / 0.33)
                          : mix(c2, c3, (x - 0.66) / 0.34);
        }}
        
        // ---------- random-feature scalar field ----------
float field(vec2 p){{
    float v     = 0.0;   // 加权后的数值累计
    float wsum  = 0.0;   // POU 权重和  Σ w_i

    for (int i = 0; i < NSUB; ++i){{
        vec2 center = texelFetch(centers_tex, ivec2(i, 0), 0).rg;
        vec2 radius = texelFetch(radii_tex,  ivec2(i, 0), 0).rg;
        vec2 s      = (p - center) / radius;          // 局部归一化坐标

        // --- 子域窗口权重 w_i(p) = ψ(s.x) · ψ(s.y) ---
        float wi = psi(s.x) * psi(s.y);
        if (wi == 0.0)           // 点不在该子域支撑内
            continue;

        // --- 该子域的随机特征网络输出 f_i(p) ---
        float sub = 0.0;
        for (int j = 0; j < NHIDDEN; ++j){{
            vec2  w = texelFetch(weights_tex, ivec2(j, i), 0).rg;
            float b = texelFetch(biases_tex,  ivec2(j, i), 0).r;
            float o = texelFetch(outputW_tex, ivec2(j, i), 0).r;
            float a = act(dot(w, s) + b);             // 可切换激活
            sub += a * o;
        }}

        v    += wi * sub;     // 加权累加
        wsum += wi;           // 累加权重
    }}

    // -------- 归一化：Σ α_i = 1 --------
    return (wsum > 0.0) ? (v / wsum) : 0.0;
}}
        
        float sdf(vec2 p) {{
    
        return {self.glsl_sdf};
        }}
        
        void main() {{
        vec2 p = mix(bbox_min, bbox_max, (fragCoord * 0.5 + 0.5));
        if (sdf(p) > 0.0) {{
            fragColor = vec4(1.0);           // outside: white
        }} else {{
            float f = field(p);
            float t = clamp((f - vmin) / (vmax - vmin + 1e-5), 0.0, 1.0);
            fragColor = vec4(viridis(t), 1.0);
        }}
        }}
        """


class RFMVisualizer3D(RFMVisualizer):
    _VIEW_TABLE = {
        'front': (0.0, 0.0),  # +Z 方向看向原点
        'back': (math.pi, 0.0),  # -Z
        'left': (math.pi / 2, 0.0),  # +X
        'right': (-math.pi / 2, 0.0),  # -X
        'xy': (0.0, -math.pi / 2),  # 自 +Z 俯视 XY
        'xz': (0.0, 0.0),  # 与 front 同义
        'yz': (math.pi / 2, 0.0),  # +X → YZ
        'iso': (math.pi / 4, -math.asin(math.sqrt(1 / 3))),  # 经典等轴
    }

    def __init__(self, model: RFMBase, resolution=(1920, 1080), component_idx=0, view='iso', pou_type=1, act_type=0):
        super().__init__(model, resolution, component_idx)
        self.yaw, self.pitch = self._VIEW_TABLE.get(str(view).lower(), (0.0, 0.0))
        self.pou_type = pou_type  # POU type: 0=PsiA, 1=PsiB, 2=PsiBW
        self.act_type = act_type  # Activation type: 0=tanh, 1=sin, 2=cos, 3=ReLU

    def estimate_vmin_vmax(self, n_samples=1000):
        x_sampled = self.model.domain.in_sample(n_samples, with_boundary=True)
        values_sampled = self.model(x_sampled).cpu().numpy().astype('f4')[:, self.component_idx]
        return values_sampled.min(), values_sampled.max()

    def _init_gl(self):
        vert_src = """#version 330
        in vec2 in_vert;
        out vec2 fragCoord;
        void main(){ fragCoord=in_vert; gl_Position=vec4(in_vert,0.0,1.0);}"""
        frag_src = self._build_fragment_shader()
        self.ctx = moderngl.create_standalone_context()
        self.fbo = self.ctx.simple_framebuffer(self.size, components=4)
        self.fbo.use()
        self.prog = self.ctx.program(vertex_shader=vert_src, fragment_shader=frag_src)

        # ---- 标量 uniform ---- #
        self.prog['iResolution'].value = self.size
        self.prog['vmin'].value = self.vmin
        self.prog['vmax'].value = self.vmax
        self.prog['yaw'].value = float(self.yaw)
        self.prog['pitch'].value = float(self.pitch)

        # ---- 随机特征数据 ---- #
        # Texture 0 : centers  (RGB32F)  size = (n_sub, 1)
        self.centers_tex = self.ctx.texture((self.n_subdomain, 1), 3,
                                            data=self.centers.astype('f4').tobytes(), dtype='f4')
        self.centers_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.centers_tex.use(location=0)
        self.prog['centers_tex'].value = 0

        # Texture 1 : radii    (RGB32F)  size = (n_sub, 1)
        self.radii_tex = self.ctx.texture((self.n_subdomain, 1), 3,
                                          data=self.radii.astype('f4').tobytes(), dtype='f4')
        self.radii_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.radii_tex.use(location=1)
        self.prog['radii_tex'].value = 1

        # Texture 2 : weights  (RGB32F)  size = (n_hidden, n_sub)
        w_tex = self.weights.transpose(0, 2, 1).astype('f4')  # (n_sub, n_hidden, 3)
        self.weights_tex = self.ctx.texture((self.n_hidden, self.n_subdomain), 3,
                                            data=w_tex.tobytes(), dtype='f4')
        self.weights_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.weights_tex.use(location=2)
        self.prog['weights_tex'].value = 2

        # Texture 3 : biases   (R32F)    size = (n_hidden, n_sub)
        b_tex = self.biases.T.astype('f4')  # (n_hidden, n_sub)
        self.biases_tex = self.ctx.texture((self.n_hidden, self.n_subdomain), 1,
                                           data=b_tex.tobytes(), dtype='f4')
        self.biases_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.biases_tex.use(location=3)
        self.prog['biases_tex'].value = 3

        # Texture 4 : outputW  (R32F)    size = (n_hidden, n_sub)
        o_tex = self.outputW.reshape(self.n_subdomain, self.n_hidden).T.astype('f4')
        self.outputW_tex = self.ctx.texture((self.n_hidden, self.n_subdomain), 1,
                                            data=o_tex.tobytes(), dtype='f4')
        self.outputW_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.outputW_tex.use(location=4)
        self.prog['outputW_tex'].value = 4

        # Full-screen quad
        quad = self.ctx.buffer(np.array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1], 'f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, quad, 'in_vert')

    def _build_fragment_shader(self):
        # GLSL fragment shader for 3D SDF visualization
        return f"""
        #version 330
        
        #define DIM       {self.dim}           // 2=2-D, 3=3-D ……
        #define NSUB      {self.n_subdomain}           // 子域数
        #define NHIDDEN   {self.n_hidden}          // 每子域隐藏元
        #define POU_TYPE  {self.pou_type}           // 0=PsiA, 1=PsiB, 2=PsiBW …
        #define ACT_TYPE  {self.act_type}           // 0=tanh, 1=sin, 2=cos, 3=ReLU
        in  vec2 fragCoord;
        out vec4 fragColor;

        uniform vec2  iResolution;
        uniform float yaw;
        uniform float pitch;

        uniform sampler2D centers_tex;   // (i,0) → vec3 center
        uniform sampler2D radii_tex;     // (i,0) → vec3 radius
        uniform sampler2D weights_tex;   // (j,i) → vec3 weight
        uniform sampler2D biases_tex;    // (j,i) → float bias
        uniform sampler2D outputW_tex;   // (j,i) → float output weight
        uniform float vmin;
        uniform float vmax;
        
        float psiA(float x){{ return abs(x)<=1.0 ? 1.0 : 0.0; }}

        float psiB(float x){{
            return (x<-1.25)?0.0:
                   (x<-0.75)?0.5*(1.0+sin(2.0*3.14159265*x)):
                   (x<=0.75)?1.0:
                   (x<=1.25)?0.5*(1.0-sin(2.0*3.14159265*x)):
                              0.0;
        }}
        
        float psi(float x){{
        #if   POU_TYPE == 0
            return psiA(x);
        #elif POU_TYPE == 1
            return psiB(x);
        #else                // 例如 PsiBW
            return psiB(x);
        #endif
        }}
        
        float act(float z){{
        #if   ACT_TYPE == 0
            return tanh(z);
        #elif ACT_TYPE == 1
            return sin(z);
        #elif ACT_TYPE == 2
            return cos(z);
        #else                       // ReLU
            return max(z, 0.0);
        #endif
        }}

        // ---------- viridis ----------
        vec3 viridis(float x){{
            x=clamp(x,0.0,1.0);
            const vec3 c0=vec3(0.267004,0.004874,0.329415);
            const vec3 c1=vec3(0.190631,0.407061,0.556089);
            const vec3 c2=vec3(0.20803,0.718701,0.472873);
            const vec3 c3=vec3(0.993248,0.906157,0.143936);
            return (x<0.33)?mix(c0,c1,x/0.33)
                 :(x<0.66)?mix(c1,c2,(x-0.33)/0.33)
                          :mix(c2,c3,(x-0.66)/0.34);
        }}


        // ---------- 3-D random-feature scalar field ----------
        float computeField(vec3 p){{
            float v    = 0.0;   // 加权数值累加  Σ w_i f_i
            float wsum = 0.0;   // 单位分解权重和  Σ w_i
        
            for (int i = 0; i < NSUB; ++i){{
                vec3 center = texelFetch(centers_tex, ivec2(i, 0), 0).rgb;
                vec3 radius = texelFetch(radii_tex,  ivec2(i, 0), 0).rgb;
                vec3 s      = (p - center) / radius;              // 局部归一化坐标
        
                // ---- 子域窗口权重 w_i(p) = ψ(s.x)·ψ(s.y)·ψ(s.z) ----
                float wi = psi(s.x) * psi(s.y) * psi(s.z);
                if (wi == 0.0)               // 点不在该子域支撑内
                    continue;
        
                // ---- 该子域的随机特征网络输出 f_i(p) ----
                float sub = 0.0;
                for (int j = 0; j < NHIDDEN; ++j){{
                    vec3  w = texelFetch(weights_tex, ivec2(j, i), 0).rgb;
                    float b = texelFetch(biases_tex,  ivec2(j, i), 0).r;
                    float o = texelFetch(outputW_tex, ivec2(j, i), 0).r;
                    float a = act(dot(w, s) + b);   // 可切换激活函数
                    sub += a * o;
                }}
        
                v    += wi * sub;     // 加权累加
                wsum += wi;           // 累加权重
            }}
        
            // -------- 归一化：Σ α_i = 1 --------
            return (wsum > 0.0) ? (v / wsum) : 0.0;
        }}

        float sceneSDF(vec3 p){{return {self.glsl_sdf()};}};

        vec3 estimateNormal(vec3 p){{
            const float e=0.001;
            return normalize(vec3(
                sceneSDF(p+vec3(e,0,0))-sceneSDF(p-vec3(e,0,0)),
                sceneSDF(p+vec3(0,e,0))-sceneSDF(p-vec3(0,e,0)),
                sceneSDF(p+vec3(0,0,e))-sceneSDF(p-vec3(0,0,e))
            ));
        }}

        #define MAX_STEPS 128
        #define EPSILON   0.001
        #define FAR       100.0

        void main(){{
            // uv in [-1,1] with aspect correction
            vec2 uv=(fragCoord*0.5+0.5)*iResolution;
            uv=(uv-0.5*iResolution)/iResolution.y;

            // build view rotation (same公式与前 3D 交互版本一致)
            float cy=cos(yaw), sy=sin(yaw);
            float cx=cos(pitch), sx=sin(pitch);
            mat3 view=mat3(
                 cy,  sx*sy, -cx*sy,
                 0.0, cx,     sx,
                 sy, -sx*cy,  cx*cy
            );

            vec3 ro=view*vec3(0.0,0.0,2.5);
            vec3 rd=normalize(view*vec3(uv,-1.0));

            float t=0.0;
            vec3 col=vec3(1.0);

            for(int i=0;i<MAX_STEPS;++i){{
                vec3 p=ro+rd*t;
                float d=sceneSDF(p);
                if(d<EPSILON){{
                    vec3 n=estimateNormal(p);
                    float diff=max(dot(n,normalize(vec3(0.8,0.6,0.2))),0.0);

                    float f=computeField(p);
                    float tt=clamp((f-vmin)/(vmax-vmin+1e-5),0.0,1.0);
                    vec3 base=viridis(tt);

                    col=mix(base,vec3(1.0),0.2)*diff+base*0.3;
                    break;
                }}
                if(t>FAR)break;
                t+=d;
            }}
            fragColor=vec4(col,1.0);
        }}
        """

    def plot(self):
        self._init_gl()
        self.fbo.clear()
        self.vao.render()
        data = self.fbo.read(components=3)
        img = Image.frombytes('RGB', self.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        np_img = np.asarray(img) / 255.0
        fig, ax = plt.subplots()
        ax.imshow(np_img, origin='lower')
        sm = plt.cm.ScalarMappable(cmap='viridis',
                                   norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
        sm.set_array([])
        plt.colorbar(sm, ax=ax).set_label('Scalar Field Value')
        ax.set_axis_off()
        plt.tight_layout()
