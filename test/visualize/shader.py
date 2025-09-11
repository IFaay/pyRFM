import moderngl
import moderngl_window as mglw
from moderngl_window import geometry
import math
import numpy as np
from moderngl_window.integrations.imgui import ModernglWindowRenderer
import imgui
import sys


# -----------------------------
# 3D SDF Raymarching App (Viridis colormap)
# -----------------------------
class SDFShaderApp3D(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "SDF Raymarching with Viridis"
    window_size = (1920, 1080)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.quad = geometry.quad_fs()
        # Create shader program for raymarching
        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                out vec2 fragCoord;
                void main() {
                    fragCoord = in_position;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            ''',
            fragment_shader=self.fragment_shader_code()
        )
        # Camera and mouse state
        self.mouse_pos = (0, 0)
        self.mouse_down = False
        self.yaw = 0.0
        self.pitch = 0.0
        self.last_mouse_pos = None

    def fragment_shader_code(self):
        # Fragment shader for SDF raymarching and Viridis colormap
        return '''
            #version 330
            in vec2 fragCoord;
            out vec4 fragColor;
            uniform vec2 iResolution;
            uniform vec4 iMouse;
            uniform float iTime;
            float t = iTime;
            const int MAX_STEPS = 128;
            const float EPSILON = 0.001;
            const float FAR = 100.0;
            // Scalar field function
            float computeField(vec3 p) {
                return sin(p.x * 3.0 + p.y * 2.0 + p.z * 4.0);
            }
            // Torus SDF (distance function)
            float torusSDF(vec3 p) {
                return max(length(vec2(dot(p, vec3(-0.000000,0.000000,1.000000)), dot(p, vec3(-0.707107,0.707107,-0.000000))) - vec2(0.0, 0.0)) - 0.5, abs(dot(p, vec3(-0.707107,-0.707107,0.000000))) - 0.707107);
            }
            // Scene SDF
            float sceneSDF(vec3 p) {
                return torusSDF(p);
            }
            // Estimate normal for shading
            vec3 estimateNormal(vec3 p) {
                float e = EPSILON;
                return normalize(vec3(
                    sceneSDF(p + vec3(e, 0, 0)) - sceneSDF(p - vec3(e, 0, 0)),
                    sceneSDF(p + vec3(0, e, 0)) - sceneSDF(p - vec3(0, e, 0)),
                    sceneSDF(p + vec3(0, 0, e)) - sceneSDF(p - vec3(0, 0, e))
                ));
            }
            // Viridis colormap
            vec3 viridisColor(float x) {
                x = clamp(x, 0.0, 1.0);
                vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
                vec3 c1 = vec3(0.190631, 0.407061, 0.556089);
                vec3 c2 = vec3(0.20803, 0.718701, 0.472873);
                vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
                if (x < 0.33) {
                    return mix(c0, c1, x / 0.33);
                } else if (x < 0.66) {
                    return mix(c1, c2, (x - 0.33) / 0.33);
                } else {
                    return mix(c2, c3, (x - 0.66) / 0.34);
                }
            }
            // Estimate min/max for colormap normalization
            void estimateMinMax(out float minVal, out float maxVal) {
                minVal =  10000.0;
                maxVal = -10000.0;
                for (int xi = 0; xi < 5; xi++)
                for (int yi = 0; yi < 5; yi++)
                for (int zi = 0; zi < 5; zi++) {
                    vec3 p = vec3(
                        mix(-1.0, 1.0, float(xi) / 4.0),
                        mix(-1.0, 1.0, float(yi) / 4.0),
                        mix(-1.0, 1.0, float(zi) / 4.0)
                    );
                    float val = computeField(p);
                    minVal = min(minVal, val);
                    maxVal = max(maxVal, val);
                }
            }
            // Main raymarching loop
            void main() {
                vec2 uv = (fragCoord * 0.5 + 0.5) * iResolution;
                uv = (uv - 0.5 * iResolution) / iResolution.y;
                vec3 ro = vec3(0.0, 0.0, 2.5);
                vec3 rd = normalize(vec3(uv, -1.0));
                float yaw = iMouse.x;
                float pitch = iMouse.y;
                float cy = cos(yaw), sy = sin(yaw);
                float cx = cos(pitch), sx = sin(pitch);
                mat3 view = mat3(
                    cy, sx * sy, -cx * sy,
                    0.0, cx, sx,
                    sy, -sx * cy, cx * cy
                );
                ro = view * ro;
                rd = view * rd;
                float minVal, maxVal;
                estimateMinMax(minVal, maxVal);
                float range = maxVal - minVal + 1e-5;
                float t = 0.0;
                vec3 col = vec3(0.0);
                vec3 bg = vec3(1.0);
                bool hit = false;
                for (int i = 0; i < MAX_STEPS; ++i) {
                    vec3 p = ro + rd * t;
                    float d = sceneSDF(p);
                    if (d < EPSILON) {
                        hit = true;
                        vec3 lightDir = normalize((ro + vec3(0.0, 1.0, 0.0)) - p);
                        vec3 normal = estimateNormal(p);
                        vec3 viewDir = normalize(-rd);
                        float diff = max(dot(normal, lightDir), 0.0);
                        float bounce = max(dot(normal, vec3(0.0, -1.0, 0.0)), 0.0) * 0.2;
                        vec3 halfDir = normalize(lightDir + viewDir);
                        float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);
                        float val = computeField(p);
                        val = clamp((val - minVal) / range, 0.0, 1.0);
                        vec3 color = viridisColor(val);
                        vec3 base = mix(color, vec3(1.0), -0.4);
                        vec3 highlight = vec3(1.0) * pow(spec, 10.0);
                        vec3 glow = vec3(1.0) * 0.08 * smoothstep(0.4, 1.0, diff);
                        vec3 refraction = mix(base * 0.85 + highlight, bg, 0.2) + glow;
                        float shadow = smoothstep(0.0, 0.3, dot(normal, lightDir));
                        vec3 tint = vec3(0.9, 0.95, 1.0);
                        col = refraction * (0.6 * diff * shadow)
                            + refraction * 0.6
                            + refraction * bounce * 0.8
                            + vec3(1.0) * spec * 0.1 * tint;
                        break;
                    }
                    if (t > FAR) break;
                    t += d;
                }
                fragColor = vec4(hit ? col : bg, 1.0);
            }
        '''

    def on_render(self, time: float, frame_time: float):
        # Render loop: auto-rotate camera for showcase
        self.ctx.clear(0.0, 0.0, 0.0)
        self.yaw = time * 0.5  # 360° yaw every ~12.6s
        self.pitch = math.sin(time * 0.25) * 1.5  # pitch swing ±1.5 rad
        self.program['iResolution'].value = self.window_size
        self.program['iMouse'].value = (float(self.yaw), float(self.pitch), 0.0, 0.0)
        self.program['iTime'].value = time
        self.quad.render(self.program)

    def mouse_position_event(self, x, y, dx, dy):
        # Mouse movement for camera rotation
        self.mouse_pos = (x, self.window_size[1] - y)
        if self.mouse_down and self.last_mouse_pos:
            dx = x - self.last_mouse_pos[0]
            dy = y - self.last_mouse_pos[1]
            self.yaw += dx * 0.01
            self.pitch += dy * 0.01
            self.pitch = max(-1.57, min(1.57, self.pitch))
            self.last_mouse_pos = (x, y)

    def mouse_press_event(self, x, y, button):
        # Mouse press event
        self.mouse_down = True
        self.last_mouse_pos = (x, y)

    def mouse_release_event(self, x, y, button):
        # Mouse release event
        self.mouse_down = False
        self.last_mouse_pos = None

    def mouse_drag_event(self, x, y, dx, dy, buttons):
        # Mouse drag for camera rotation (pyglet backend)
        self.yaw += dx * 0.01
        self.pitch += dy * 0.01
        self.pitch = max(-1.57, min(1.57, self.pitch))
        self.mouse_pos = (x, self.window_size[1] - y)


# -----------------------------
# 2D SDF Visualization App (Viridis colormap)
# -----------------------------
class SDFShaderApp2D(mglw.WindowConfig):
    """简单二维 SDF 可视化（Viridis 颜色图）"""
    gl_version = (3, 3)
    title = "2-D SDF with Viridis"
    window_size = (1920, 1080)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # --- random feature parameters --- #
        self.n_sub = 2
        self.n_hidden = 2
        self.dim = 2  # 2‑D

        centers = np.array([[0.0, 0.0],
                            [0.1, 0.1]], dtype='f4')
        radii = np.array([[1.0, 1.0],
                          [1.0, 1.0]], dtype='f4')
        weights = np.random.randn(self.n_sub, self.dim, self.n_hidden).astype('f4')
        biases = np.random.randn(self.n_sub, self.n_hidden).astype('f4')
        outputW = np.random.randn(self.n_sub * self.n_hidden).astype('f4')

        self.bbox_min = np.array([-0.4, -0.4], dtype='f4')
        self.bbox_max = np.array([0.4, 0.4], dtype='f4')

        self.quad = geometry.quad_fs()

        self.program = self.ctx.program(
            vertex_shader='''
                #version 330
                in vec2 in_position;
                out vec2 fragCoord;
                void main() {
                    fragCoord = in_position;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            ''',
            fragment_shader=self.build_fragment_shader()
        )

        # set resolution once (window is not resizable in this example)
        self.program['iResolution'].value = self.window_size

        # upload random‑feature data
        self.program['centers'].write(centers.tobytes())
        self.program['radii'].write(radii.tobytes())
        self.program['weights'].write(weights.reshape(-1).tobytes())
        self.program['biases'].write(biases.reshape(-1).tobytes())
        self.program['outputW'].write(outputW.tobytes())

        self.program['bbox_min'].value = tuple(self.bbox_min)
        self.program['bbox_max'].value = tuple(self.bbox_max)

        # Ensure ImGui context exists before initializing the renderer
        imgui.create_context()
        imgui.set_current_context(imgui.get_current_context())
        io = imgui.get_io()
        io.font_global_scale = max(1.5, min(4.0, self.window_size[0] / 800.0))
        self.imgui = ModernglWindowRenderer(self.wnd)

    def build_fragment_shader(self):
        # Build fragment shader for 2D SDF visualization with Viridis colormap
        return f'''
        #version 330
        in vec2 fragCoord;
        out vec4 fragColor;

        uniform vec2 iResolution;
        vec2 scale = iResolution;

        uniform vec2  centers[{self.n_sub}];
        uniform vec2  radii[{self.n_sub}];
        uniform float weights[{self.n_sub * self.dim * self.n_hidden}];
        uniform float biases[{self.n_sub * self.n_hidden}];
        uniform float outputW[{self.n_sub * self.n_hidden}];

        uniform vec2 bbox_min;
        uniform vec2 bbox_max;

        // ------------ viridis ------------- //
        vec3 viridisColor(float x) {{
            x = clamp(x, 0.0, 1.0);
            const vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
            const vec3 c1 = vec3(0.190631, 0.407061, 0.556089);
            const vec3 c2 = vec3(0.20803, 0.718701, 0.472873);
            const vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
            if (x < 0.33) {{
                return mix(c0, c1, x / 0.33);
            }} else if (x < 0.66) {{
                return mix(c1, c2, (x - 0.33) / 0.33);
            }} else {{
                return mix(c2, c3, (x - 0.66) / 0.34);
            }}
        }}

        // ---------- random feature scalar field ---------- //
        float computeField(vec2 p) {{
            float val = 0.0;
            for (int i = 0; i < {self.n_sub}; ++i) {{
                vec2 s = (p - centers[i]) / radii[i];
                for (int j = 0; j < {self.n_hidden}; ++j) {{
                    int idx = i * {self.dim} * {self.n_hidden} + j * {self.dim};
                    vec2 w = vec2(weights[idx], weights[idx + 1]);
                    float b = biases[i * {self.n_hidden} + j];
                    float a = tanh(dot(w, s) + b);
                    val += a * outputW[i * {self.n_hidden} + j];
                }}
            }}
            return val;
        }}

        void computeMinMax(out float v_min, out float v_max) {{
            v_min = 10000.0;
            v_max = -10000.0;
            for (int xi = 0; xi < 10; ++xi)
            for (int yi = 0; yi < 10; ++yi) {{
                vec2 sp = vec2(
                    mix(bbox_min.x, bbox_max.x, float(xi) / 4.0),
                    mix(bbox_min.y, bbox_max.y, float(yi) / 4.0)
                );
                float val = computeField(sp);
                v_min = min(v_min, val);
                v_max = max(v_max, val);
            }}
        }}

        // simple 2‑D SDF: circle of radius 0.5
        float sdf(vec2 p) {{          // 一个圆
            vec2 b = vec2(0.35);
            float box = length(max(abs(p - vec2(0.0, 0.0)) - b, 0.0));
            return min(length(p) - 0.4, box);  // 半径 0.4
        }}

        void main() {{
            // map fragCoord from [-1,1] quad to pixel space, then to bbox space
            vec2 p = mix(bbox_min, bbox_max, (fragCoord * 0.5 + 0.5));

            float d = sdf(p);

            if (d > 0.0) {{
                fragColor = vec4(1.0);   // outside: white
            }} else {{
                float f = computeField(p);
                float v_min, v_max;
                computeMinMax(v_min, v_max);
                float t = clamp((f - v_min) / (v_max - v_min + 1e-5), 0.0, 1.0);
                fragColor = vec4(viridisColor(t), 1.0);
            }}
        }}
        '''

    def on_render(self, time: float, frame_time: float):
        # Viewport adjustment and rendering
        self.ctx.viewport = (
            int(self.window_size[0] * (0.5 - 0.5 * (self.bbox_max[0] - self.bbox_min[0]))),
            int(self.window_size[1] * (0.5 - 0.5 * (self.bbox_max[1] - self.bbox_min[1]))),
            int(self.window_size[0] * (self.bbox_max[0] - self.bbox_min[0])),
            int(self.window_size[1] * (self.bbox_max[1] - self.bbox_min[1]))
        )
        self.ctx.clear(1.0, 1.0, 1.0)
        self.program['iResolution'].value = self.window_size
        self.quad.render(self.program)

        # Capture frame and use matplotlib for annotation
        from PIL import Image
        import numpy as np
        import matplotlib.pyplot as plt

        raw = self.ctx.screen.read(components=3, alignment=1)
        img = Image.frombytes('RGB', self.window_size, raw).transpose(Image.FLIP_TOP_BOTTOM)
        np_img = np.array(img) / 255.0

        fig, ax = plt.subplots(figsize=(8, 8))
        im = ax.imshow(np_img, extent=[self.bbox_min[0], self.bbox_max[0], self.bbox_min[1], self.bbox_max[1]],
                       origin='lower')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('SDF Field with Viridis')
        ax.grid(True)
        plt.tight_layout()
        plt.show()
        sys.exit()

    def on_resize(self, width: int, height: int):
        if hasattr(self, 'imgui'):
            self.imgui.resize(width, height)


# --- Off-screen 2D SDF renderer (class version) --- #
import matplotlib.pyplot as plt
from PIL import Image


class SDFShaderApp2DOffscreen:
    """Off‑screen 2‑D SDF renderer that keeps a class interface."""

    def __init__(self,
                 size=(1920, 1080),
                 bbox_min=(-0.5, -0.5),
                 bbox_max=(0.5, 0.5),
                 n_sub=2, n_hidden=2, dim=2):
        self.size = size
        self.bbox_min = bbox_min
        self.bbox_max = bbox_max

        self.n_sub = n_sub
        self.n_hidden = n_hidden
        self.dim = dim
        # Random features
        self.centers = np.array([[0.0, 0.0], [0.1, 0.1]], dtype='f4')
        self.radii = np.array([[1.0, 1.0], [1.0, 1.0]], dtype='f4')
        self.weights = np.random.randn(n_sub, dim, n_hidden).astype('f4')
        self.biases = np.random.randn(n_sub, n_hidden).astype('f4')
        self.outputW = np.random.randn(n_sub * n_hidden).astype('f4')
        # Estimate vmin/vmax on CPU
        self.vmin, self.vmax = self.estimate_vmin_vmax()

        # Build GL objects
        self._init_gl()

    def estimate_vmin_vmax(self, N=32):
        vmin, vmax = float('inf'), float('-inf')
        for xi in range(N):
            for yi in range(N):
                x = np.interp(xi, [0, N - 1], [self.bbox_min[0], self.bbox_max[0]])
                y = np.interp(yi, [0, N - 1], [self.bbox_min[1], self.bbox_max[1]])
                p = (np.array([x, y]) - self.centers) / self.radii
                val = 0.0
                for i in range(self.n_sub):
                    for j in range(self.n_hidden):
                        w = self.weights[i, :, j]
                        b = self.biases[i, j]
                        a = np.tanh(np.dot(w, p[i]) + b)
                        val += a * self.outputW[i * self.n_hidden + j]
                vmin = min(vmin, val)
                vmax = max(vmax, val)
        return vmin, vmax

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
        self.centers_tex = self.ctx.texture((self.n_sub, 1), 2, data=self.centers.astype('f4').tobytes(), dtype='f4')
        self.centers_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.centers_tex.use(location=0)
        self.prog['centers_tex'].value = 0

        # Texture 1: radii (vec2), shape = (n_sub, 1)
        self.radii_tex = self.ctx.texture((self.n_sub, 1), 2, data=self.radii.astype('f4').tobytes(), dtype='f4')
        self.radii_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.radii_tex.use(location=1)
        self.prog['radii_tex'].value = 1

        # Texture 2: weights (vec2), layout = (row = subfeature, col = hidden neuron)
        w_tex_data = self.weights.transpose(0, 2, 1).astype('f4')  # shape (n_sub, n_hidden, 2)
        self.weights_tex = self.ctx.texture((self.n_hidden, self.n_sub), 2, data=w_tex_data.tobytes(), dtype='f4')
        self.weights_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.weights_tex.use(location=2)
        self.prog['weights_tex'].value = 2

        # Texture 3: biases (float), shape = (n_hidden, n_sub)
        self.biases_tex = self.ctx.texture((self.n_hidden, self.n_sub), 1, data=self.biases.astype('f4').tobytes(),
                                           dtype='f4')
        self.biases_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.biases_tex.use(location=3)
        self.prog['biases_tex'].value = 3

        # Texture 4: output weights (float), shape = (n_hidden, n_sub)
        outw = self.outputW.reshape(self.n_sub, self.n_hidden).astype('f4')
        self.outputW_tex = self.ctx.texture((self.n_hidden, self.n_sub), 1, data=outw.tobytes(), dtype='f4')
        self.outputW_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.outputW_tex.use(location=4)
        self.prog['outputW_tex'].value = 4

        self.prog['vmin'].value = self.vmin
        self.prog['vmax'].value = self.vmax
        quad = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1], 'f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, quad, 'in_vert')

    def _build_fragment_shader(self):
        ns = self.n_sub;
        nh = self.n_hidden;
        d = self.dim
        # GLSL fragment shader for 2D SDF visualization
        return f"""
        #version 330
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
        float field(vec2 p) {{
            float v = 0.0;
            for (int i = 0; i < {ns}; ++i) {{
                vec2 center = texelFetch(centers_tex, ivec2(i, 0), 0).rg;
                vec2 radius = texelFetch(radii_tex,  ivec2(i, 0), 0).rg;
                vec2 s = (p - center) / radius;
                for (int j = 0; j < {nh}; ++j) {{
                    vec2 w = texelFetch(weights_tex, ivec2(j, i), 0).rg;
                    float b = texelFetch(biases_tex,  ivec2(j, i), 0).r;
                    float o = texelFetch(outputW_tex, ivec2(j, i), 0).r;
                    float a = tanh(dot(w, s) + b);
                    v += a * o;
                }}
            }}
            return v;
        }}

        float sdf(vec2 p) {{
            // circle (r=0.4) union small box
            return min(length(max(abs(p - vec2(0.0,0.0)) - vec2(0.1,0.1), 0.0))+ min(max(abs(p.x-0.0)-0.1, abs(p.y-0.0)-0.1), 0.0), length(p - vec2(0.5, 0.5)) - 0.5);
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

    def render(self, show=True, save_path=None):
        # Off-screen rendering and optional saving/displaying
        self.fbo.clear()
        self.vao.render()
        data = self.fbo.read(components=3)
        img = Image.frombytes('RGB', self.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        if save_path:
            img.save(save_path)
        if show:
            np_img = np.asarray(img) / 255.0
            fig, ax = plt.subplots()
            ax.imshow(np_img, extent=[self.bbox_min[0], self.bbox_max[0],
                                      self.bbox_min[1], self.bbox_max[1]], origin='lower')
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label('Scalar Field Value')
            plt.tight_layout()
            ax.set_xlabel('x');
            ax.set_ylabel('y');
            ax.set_title('SDF Field (Viridis)')
            # ax.grid(True);
            plt.tight_layout();
            plt.show()
        return img


# --- 3D Off‑screen SDF renderer (class version) --- #
class SDFShaderApp3DOffscreen:
    """Off-screen 3-D SDF renderer with a 3-D random-feature scalar field and Viridis colormap.
       view ∈ {'front','back','left','right','xy','xz','yz','iso'} 决定初始朝向。
    """

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

    def __init__(self,
                 size=(1920, 1080),
                 bbox_min=(-1.0, -1.0, -1.0),
                 bbox_max=(1.0, 1.0, 1.0),
                 n_sub=2, n_hidden=2, dim=3,
                 view='iso'):
        self.size = size
        self.bbox_min = np.array(bbox_min, dtype='f4')
        self.bbox_max = np.array(bbox_max, dtype='f4')
        self.n_sub = n_sub
        self.n_hidden = n_hidden
        self.dim = dim

        # --------- 视角 --------- #
        self.yaw, self.pitch = self._VIEW_TABLE.get(str(view).lower(), (0.0, 0.0))

        # --- random-feature parameters --- #
        self.centers = np.random.uniform(-0.3, 0.3, size=(n_sub, dim)).astype('f4')
        self.radii = np.ones((n_sub, dim), dtype='f4')
        self.weights = np.random.randn(n_sub, dim, n_hidden).astype('f4')
        self.biases = np.random.randn(n_sub, n_hidden).astype('f4')
        self.outputW = np.random.randn(n_sub * n_hidden).astype('f4')

        # Estimate scalar-field range on CPU for colormap normalization
        self.vmin, self.vmax = self._estimate_vmin_vmax()
        self._init_gl()

    # ---------- CPU helpers ---------- #
    def _estimate_vmin_vmax(self, N=16):
        vmin, vmax = float('inf'), float('-inf')
        grid = np.linspace(0, 1, N)
        for xi in grid:
            for yi in grid:
                for zi in grid:
                    p = self.bbox_min + np.array([xi, yi, zi]) * (self.bbox_max - self.bbox_min)
                    val = 0.0
                    for i in range(self.n_sub):
                        s = (p - self.centers[i]) / self.radii[i]
                        for j in range(self.n_hidden):
                            w = self.weights[i, :, j]
                            b = self.biases[i, j]
                            a = np.tanh(np.dot(w, s) + b)
                            val += a * self.outputW[i * self.n_hidden + j]
                    vmin = min(vmin, val)
                    vmax = max(vmax, val)
        return vmin, vmax

    # ---------- OpenGL setup ---------- #
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
        self.centers_tex = self.ctx.texture((self.n_sub, 1), 3,
                                            data=self.centers.astype('f4').tobytes(), dtype='f4')
        self.centers_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.centers_tex.use(location=0)
        self.prog['centers_tex'].value = 0

        # Texture 1 : radii    (RGB32F)  size = (n_sub, 1)
        self.radii_tex = self.ctx.texture((self.n_sub, 1), 3,
                                          data=self.radii.astype('f4').tobytes(), dtype='f4')
        self.radii_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.radii_tex.use(location=1)
        self.prog['radii_tex'].value = 1

        # Texture 2 : weights  (RGB32F)  size = (n_hidden, n_sub)
        w_tex = self.weights.transpose(0, 2, 1).astype('f4')  # (n_sub, n_hidden, 3)
        self.weights_tex = self.ctx.texture((self.n_hidden, self.n_sub), 3,
                                            data=w_tex.tobytes(), dtype='f4')
        self.weights_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.weights_tex.use(location=2)
        self.prog['weights_tex'].value = 2

        # Texture 3 : biases   (R32F)    size = (n_hidden, n_sub)
        b_tex = self.biases.T.astype('f4')  # (n_hidden, n_sub)
        self.biases_tex = self.ctx.texture((self.n_hidden, self.n_sub), 1,
                                           data=b_tex.tobytes(), dtype='f4')
        self.biases_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.biases_tex.use(location=3)
        self.prog['biases_tex'].value = 3

        # Texture 4 : outputW  (R32F)    size = (n_hidden, n_sub)
        o_tex = self.outputW.reshape(self.n_sub, self.n_hidden).T.astype('f4')
        self.outputW_tex = self.ctx.texture((self.n_hidden, self.n_sub), 1,
                                            data=o_tex.tobytes(), dtype='f4')
        self.outputW_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.outputW_tex.use(location=4)
        self.prog['outputW_tex'].value = 4

        # Full-screen quad
        quad = self.ctx.buffer(np.array([
            -1, -1, 1, -1, -1, 1,
            -1, 1, 1, -1, 1, 1], 'f4').tobytes())
        self.vao = self.ctx.simple_vertex_array(self.prog, quad, 'in_vert')

    # ---------- GLSL fragment shader ---------- #
    def _build_fragment_shader(self):
        ns = self.n_sub
        nh = self.n_hidden
        dim = self.dim
        # GLSL fragment shader for 3D SDF visualization
        return f"""
        #version 330
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
        float val = 0.0;
        for(int i = 0; i < {self.n_sub}; ++i){{
            vec3 center = texelFetch(centers_tex, ivec2(i, 0), 0).rgb;
            vec3 radius = texelFetch(radii_tex,  ivec2(i, 0), 0).rgb;
            vec3 s = (p - center) / radius;
            for(int j = 0; j < {self.n_hidden}; ++j){{
                vec3  w = texelFetch(weights_tex, ivec2(j, i), 0).rgb;
                float b = texelFetch(biases_tex,  ivec2(j, i), 0).r;
                float o = texelFetch(outputW_tex, ivec2(j, i), 0).r;
                float a = tanh(dot(w, s) + b);
                val += a * o;
            }}
        }}
        return val;
    }}

        // ---------- scene SDF ----------
        float torusSDF(vec3 p){{
            
            return max(length(vec2(dot(p, vec3(0.000000,0.000000,-1.000000)), dot(p, vec3(-0.000000,-1.000000,-0.000000))) - vec2(0.0, 0.0)) - 0.5, abs(dot(p, vec3(-1.000000,0.000000,0.000000))) - 3.000000);
}}
        float sceneSDF(vec3 p){{return torusSDF(p);}};

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

    # ---------- public render ---------- #
    def render(self, show=True, save_path=None):
        # Off-screen rendering and optional saving/displaying
        self.fbo.clear()
        self.vao.render()
        data = self.fbo.read(components=3)
        img = Image.frombytes('RGB', self.size, data).transpose(Image.FLIP_TOP_BOTTOM)
        if save_path: img.save(save_path)
        if show:
            np_img = np.asarray(img) / 255.0
            fig, ax = plt.subplots()
            ax.imshow(np_img, origin='lower')
            sm = plt.cm.ScalarMappable(cmap='viridis',
                                       norm=plt.Normalize(vmin=self.vmin, vmax=self.vmax))
            sm.set_array([]);
            plt.colorbar(sm, ax=ax).set_label('Scalar Field Value')
            ax.set_axis_off();
            plt.tight_layout();
            plt.show()
        return img


if __name__ == '__main__':
    # --- Off‑screen tests (uncomment to try) -

    # SDFShaderApp2DOffscreen().render()
    SDFShaderApp3DOffscreen().render()

    # --- Interactive real‑time viewer ---
    # mglw.run_window_config(SDFShaderApp3D)
