// Signed distance for torus
float torusSDF(vec3 p) {
    vec2 q = vec2(length(p.xz) - 0.7, p.y);
    return length(q) - 0.2;
}

// Estimate normal from SDF
vec3 estimateNormal(vec3 p) {
    float eps = 0.001;
    return normalize(vec3(
        torusSDF(p + vec3(eps, 0, 0)) - torusSDF(p - vec3(eps, 0, 0)),
        torusSDF(p + vec3(0, eps, 0)) - torusSDF(p - vec3(0, eps, 0)),
        torusSDF(p + vec3(0, 0, eps)) - torusSDF(p - vec3(0, 0, eps))
    ));
}

// Jet colormap: v ∈ [0,1]
vec3 jetColor(float v) {
    v = clamp(v, 0.0, 1.0);
    return clamp(vec3(
        1.5 - abs(4.0 * v - 3.0),
        1.5 - abs(4.0 * v - 2.0),
        1.5 - abs(4.0 * v - 1.0)
    ), 0.0, 1.0);
}

// Approximate AO using SDF sampling
float fakeAO(vec3 p, vec3 n) {
    float ao = 0.0;
    float sca = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.02 * float(i);
        float d = torusSDF(p + n * h);
        ao += (h - d) * sca;
        sca *= 0.7;
    }
    return clamp(1.0 - ao, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 iResolution = iResolution.xy;
    vec4 iMouse = iMouse;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Camera
    vec3 ro = vec3(0.0, 0.0, 2.5);
    vec3 rd = normalize(vec3(uv, -1.0));

    // Mouse-controlled rotation
    float yaw = (iMouse.z > 0.0 ? iMouse.x : iMouse.x) / iResolution.x * 6.2831;
    float pitch = (iMouse.z > 0.0 ? iMouse.y : iMouse.y) / iResolution.y * 3.1416 - 1.57;
    float cy = cos(yaw), sy = sin(yaw);
    float cx = cos(pitch), sx = sin(pitch);
    mat3 view = mat3(
        cy, sx * sy, -cx * sy,
        0.0, cx, sx,
        sy, -sx * cy, cx * cy
    );
    ro = view * ro;
    rd = view * rd;

    // Raymarch
    float t = 0.0;
    vec3 col = vec3(0.0);
    bool hit = false;

    for (int i = 0; i < 128; i++) {
        vec3 p = ro + rd * t;
        float d = torusSDF(p);
        if (d < 0.001) {
            hit = true;

            // Shading
            vec3 normal = estimateNormal(p);
            vec3 lightDir = normalize(vec3(0.7, 0.8, 0.9));
            vec3 viewDir = normalize(-rd);

            // Data value (you can replace with real data function)
            float val = sin(p.x * 3.0 + p.y * 4.0 + p.z * 2.0);
            val = val * 0.5 + 0.5;
            vec3 baseColor = jetColor(val);

            // Diffuse (Half-Lambert: softer)
            float diff = max(dot(normal, lightDir), 0.0);
            diff = diff * 0.5 + 0.5; // soften

            // Specular
            vec3 halfDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

            // AO
            float ao = fakeAO(p, normal);

            // Combine: baseColor modulated by AO and diffuse, high gloss specular
            col = baseColor * diff * ao * 1.1 + vec3(1.0) * spec * 0.6;
            break;
        }
        if (t > 100.0) break;
        t += d;
    }

    // Light gray background
    vec3 bg = vec3(0.985);
    fragColor = vec4(hit ? col : bg, 1.0);
}