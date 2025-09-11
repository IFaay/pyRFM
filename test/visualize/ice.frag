#define MAX_STEPS 128
#define MAX_DIST 5.0
#define SURF_DIST 0.001
#define PI 3.141592



// === Torus + Box SDF ===
float torusSDF(vec3 p) {
    vec2 q = vec2(length(p.xz) - 0.7, p.y);
    float torus = length(q) - 0.2;

    vec3 b = vec3(0.2);
    float box = length(max(abs(p - vec3(0.5, 0.0, 0.0)) - b, 0.0));

    return min(torus, box);
}

// === Viridis Color Map ===
vec3 viridisColor(float t) {
    t = clamp(t, 0.0, 1.0);
    const vec3 c0 = vec3(0.267, 0.004, 0.329);
    const vec3 c1 = vec3(0.283, 0.141, 0.458);
    const vec3 c2 = vec3(0.254, 0.265, 0.530);
    const vec3 c3 = vec3(0.207, 0.372, 0.553);
    const vec3 c4 = vec3(0.164, 0.471, 0.558);
    const vec3 c5 = vec3(0.128, 0.567, 0.551);
    const vec3 c6 = vec3(0.135, 0.659, 0.518);
    const vec3 c7 = vec3(0.267, 0.749, 0.441);
    const vec3 c8 = vec3(0.478, 0.821, 0.318);
    const vec3 c9 = vec3(0.741, 0.873, 0.150);
    const vec3 c10 = vec3(0.993, 0.906, 0.144);

    float n = t * 10.0;
    int i = int(floor(n));
    float f = fract(n);

    vec3 col;
    if (i == 0) col = mix(c0, c1, f);
    else if (i == 1) col = mix(c1, c2, f);
    else if (i == 2) col = mix(c2, c3, f);
    else if (i == 3) col = mix(c3, c4, f);
    else if (i == 4) col = mix(c4, c5, f);
    else if (i == 5) col = mix(c5, c6, f);
    else if (i == 6) col = mix(c6, c7, f);
    else if (i == 7) col = mix(c7, c8, f);
    else if (i == 8) col = mix(c8, c9, f);
    else col = mix(c9, c10, f);

    return col;
}

// === Sky Color ===
vec3 getSkyColor(vec3 rd) {
    rd.y = clamp(rd.y, 0.0, 1.0);
    return vec3(pow(1.0 - rd.y, 2.0), 1.0 - rd.y, 0.6 + (1.0 - rd.y) * 0.4) * 1.1;
}

// === Noise (seascape-style) ===
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123);
}
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f*f*(3.0 - 2.0*f);
    return mix(
        mix(hash(i + vec2(0,0)), hash(i + vec2(1,0)), u.x),
        mix(hash(i + vec2(0,1)), hash(i + vec2(1,1)), u.x),
        u.y
    );
}

// === Normal Estimation ===
vec3 estimateNormal(vec3 p) {
    float e = 0.001;
    return normalize(vec3(
        torusSDF(p + vec3(e, 0, 0)) - torusSDF(p - vec3(e, 0, 0)),
        torusSDF(p + vec3(0, e, 0)) - torusSDF(p - vec3(0, e, 0)),
        torusSDF(p + vec3(0, 0, e)) - torusSDF(p - vec3(0, 0, e))
    ));
}

// === Raymarching ===
float raymarch(vec3 ro, vec3 rd, out vec3 p) {
    float dist = 0.0;
    for (int i = 0; i < MAX_STEPS; i++) {
        p = ro + rd * dist;
        float d = torusSDF(p);
        if (d < SURF_DIST) break;
        dist += d;
        if (dist > MAX_DIST) break;
    }
    return dist;
}

// === Fresnel ===
float fresnel(vec3 n, vec3 v) {
    float f = 1.0 - max(dot(n, -v), 0.0);
    return pow(f, 3.0);
}

// === Main ===
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord.xy / iResolution.xy * 2.0 - 1.0;
    uv.x *= iResolution.x / iResolution.y;

    // Camera
    vec3 ro = vec3(0.0, 0.0, -2.5);
    vec3 rd = normalize(vec3(uv, 1.5));

    // Mouse interaction
    vec2 mouse = iMouse.xy / iResolution.xy;
    float angle = (mouse.x - 0.5) * 2.0 * PI;
    float pitch = (mouse.y - 0.5) * PI;

    mat2 rotY = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    mat2 rotX = mat2(cos(pitch), -sin(pitch), sin(pitch), cos(pitch));

    ro.xz = rotY * ro.xz;
    rd.xz = rotY * rd.xz;

    ro.yz = rotX * ro.yz;
    rd.yz = rotX * rd.yz;

    // Raymarch
    vec3 p;
    float d = raymarch(ro, rd, p);
    vec3 col;

    if (d < MAX_DIST) {
        vec3 n = estimateNormal(p);

        // Wave-like distortion
        vec3 wave = vec3(
            noise(p.yz * 4.0 + iTime),
            noise(p.zx * 4.0 + iTime + 2.0),
            noise(p.xy * 4.0 + iTime + 4.0)
        );
        n = normalize(n + (wave - 0.5) * 0.6);

        // Lighting
        vec3 lightDir = normalize(vec3(0.5, 1.0, -0.3));
        float diff = clamp(dot(n, lightDir), 0.0, 1.0);
        float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 60.0);

        // Viridis-based coloring
        float val = -sin(p.x * 3.0 + p.y * 4.0 + p.z * 3.0);
        val = val * 0.5 + 0.5;
        col = viridisColor(val);

        // Combine
        col *= 0.4 + 0.6 * diff;
        col += vec3(spec);
        col = mix(col, getSkyColor(rd), fresnel(n, rd) * 0.5);
    } else {
        col = getSkyColor(rd);
    }

    fragColor = vec4(pow(col, vec3(0.65)), 1.0);
}