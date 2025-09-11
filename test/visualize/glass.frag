// ---------- Constants ----------
const int   MAX_STEPS = 128;
const float EPSILON   = 0.001;
const float FAR       = 100.0;

// ---------- Torus + Box SDF ----------
float torusSDF(vec3 p) {
    vec2 q = vec2(length(p.xz) - 0.7, p.y);
    float torus = length(q) - 0.2;

    vec3 b = vec3(0.2);
    float box = length(max(abs(p - vec3(0.5, 0.0, 0.0)) - b, 0.0));

    return min(torus, box);
}

// ---------- Normal Estimation ----------
vec3 estimateNormal(vec3 p) {
    float e = EPSILON;
    return normalize(vec3(
        torusSDF(p + vec3(e, 0, 0)) - torusSDF(p - vec3(e, 0, 0)),
        torusSDF(p + vec3(0, e, 0)) - torusSDF(p - vec3(0, e, 0)),
        torusSDF(p + vec3(0, 0, e)) - torusSDF(p - vec3(0, 0, e))
    ));
}

// ---------- Viridis Colormap ----------
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

// ---------- Main Image ----------
void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 iRes = iResolution.xy;
    vec4 iMouseVal = iMouse;

    vec2 uv = (fragCoord - 0.5 * iRes) / iRes.y;

    // Camera setup
    vec3 ro = vec3(0.0, 0.0, 2.5);
    vec3 rd = normalize(vec3(uv, -1.0));

    float yaw = (iMouseVal.z > 0.0 ? iMouseVal.x : iMouseVal.x) / iRes.x * 6.2831;
    float pitch = (iMouseVal.z > 0.0 ? iMouseVal.y : iMouseVal.y) / iRes.y * 3.1416 - 1.57;
    float cy = cos(yaw), sy = sin(yaw);
    float cx = cos(pitch), sx = sin(pitch);
    mat3 view = mat3(
        cy, sx * sy, -cx * sy,
        0.0, cx, sx,
        sy, -sx * cy, cx * cy
    );
    ro = view * ro;
    rd = view * rd;

    // Raymarching
    float t = 0.0;
    vec3 col = vec3(0.0);
    bool hit = false;
    vec3 bg = vec3(1.0); // white background

    for (int i = 0; i < MAX_STEPS; ++i) {
        vec3 p = ro + rd * t;
        float d = torusSDF(p);
        if (d < EPSILON) {
            hit = true;

            // LIQUID GLASS MATERIAL
            vec3 lightDir = normalize((ro + vec3(0.0, 1.0, 0.0)) - p);
            vec3 normal = estimateNormal(p);
            vec3 viewDir = normalize(-rd);

            float diff = max(dot(normal, lightDir), 0.0);
            float spec = pow(max(dot(normal, normalize(lightDir + viewDir)), 0.0), 64.0);
            float fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0);
            float glow = smoothstep(0.4, 1.0, diff) * 0.1;

            float val = -sin(p.x * 3.0 + p.y * 4.0 + p.z * 3.0);
            val = val * 0.5 + 0.5;
            vec3 baseColor = viridisColor(val);

            vec3 softEdge = vec3(1.0) * pow(spec, 10.0) * 0.7;
            vec3 glass = mix(baseColor, vec3(1.0), 0.15); // milky mix
            glass += softEdge;

            vec3 refractCol = mix(bg, glass, 0.5 + 0.3 * fresnel);

            col = refractCol * (0.5 + 0.5 * diff) + glow + spec * 0.2;
            col = mix(col, vec3(1.0), 0.05 * fresnel);

            break;
        }
        if (t > FAR) break;
        t += d;
    }

    fragColor = vec4(hit ? col : bg, 1.0);
}