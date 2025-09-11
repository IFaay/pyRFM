#ifdef GL_ES
precision mediump float;
#endif

uniform vec2 u_resolution;
uniform float u_time;

float sdf(vec2 p) {
    return abs(length(p) - 0.4);
}

vec3 sdfColor(float d) {
    float scaled = clamp(d * 5.0, -1.0, 1.0);
    float t = 0.5 + 0.5 * scaled;
    return vec3(t);
}

void main() {
    vec2 uv = gl_FragCoord.xy / u_resolution;
    vec2 p = (uv - 0.5) * (u_resolution / u_resolution.y);

    float d = sdf(p);
    vec3 col = sdfColor(d);

    gl_FragColor = vec4(col, 1.0);
}
