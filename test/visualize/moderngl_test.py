import moderngl
import moderngl_window as mglw
from moderngl_window import geometry


class SDFShaderApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "SDF Circle Shader"
    window_size = (1920, 1080)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
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
            fragment_shader='''
                #version 330
                in vec2 fragCoord;
                out vec4 fragColor;

                uniform vec2 iResolution;
                uniform float dMin;

                float sdf(vec2 p) {
                    return length(p) - 0.4;
                }

                vec3 colormap_jet(float t) {
                    t = clamp(t, 0.0, 1.0);
                    float r = clamp(1.5 - abs(4.0 * t - 3.0), 0.0, 1.0);
                    float g = clamp(1.5 - abs(4.0 * t - 2.0), 0.0, 1.0);
                    float b = clamp(1.5 - abs(4.0 * t - 1.0), 0.0, 1.0);
                    return vec3(r, g, b);
                }

                vec3 sdfColor(float d) {
                    float denom = max(0.0 - dMin, 1e-6); // 避免除零
                    float t = (d - dMin) / denom;
                    return colormap_jet(t);
                }

                void main() {
                    vec2 uv = (fragCoord * 0.5 + 0.5) * iResolution;
                    vec2 p = (uv - 0.5 * iResolution) / iResolution.y;
                    float d = sdf(p);
                    if (d > 0.0) {
                        fragColor = vec4(1.0, 1.0, 1.0, 1.0); // white background
                    } else {
                        vec3 col = sdfColor(d);
                        fragColor = vec4(col, 1.0);
                    }
                }
            '''
        )
        self.program['dMin'].value = -0.4

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.0, 0.0, 0.0)
        self.program['iResolution'].value = self.window_size
        self.quad.render(self.program)


if __name__ == '__main__':
    mglw.run_window_config(SDFShaderApp)
