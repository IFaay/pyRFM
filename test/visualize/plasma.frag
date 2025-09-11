// SDF: Torus
float torusSDF(vec3 p) {
    vec2 q = vec2(length(p.xz) - 0.7, p.y);
    return length(q) - 0.2;
}

// 法线估计
vec3 estimateNormal(vec3 p) {
    float e = 0.001;
    return normalize(vec3(
        torusSDF(p + vec3(e, 0, 0)) - torusSDF(p - vec3(e, 0, 0)),
        torusSDF(p + vec3(0, e, 0)) - torusSDF(p - vec3(0, e, 0)),
        torusSDF(p + vec3(0, 0, e)) - torusSDF(p - vec3(0, 0, e))
    ));
}

// Plasma colormap: v in [0, 1]
vec3 plasmaColor(float v) {
    v = clamp(v, 0.0, 1.0);
    const vec3 c0 = vec3(0.050383, 0.029803, 0.527975);
    const vec3 c1 = vec3(0.283072, 0.125141, 0.682397);
    const vec3 c2 = vec3(0.481208, 0.194538, 0.635254);
    const vec3 c3 = vec3(0.647257, 0.248214, 0.529983);
    const vec3 c4 = vec3(0.792317, 0.313505, 0.422053);
    const vec3 c5 = vec3(0.902323, 0.453633, 0.288921);
    const vec3 c6 = vec3(0.973308, 0.621929, 0.156274);
    const vec3 c7 = vec3(0.991438, 0.749504, 0.131925);
    const vec3 c8 = vec3(0.940015, 0.894855, 0.217772);
    const vec3 c9 = vec3(0.816333, 0.977571, 0.350728);
    const vec3 c10 = vec3(0.678489, 0.998365, 0.444000);

    float n = v * 10.0;
    int i = int(floor(n));
    float f = fract(n);

    vec3 a = vec3(0.0);
    vec3 b = vec3(0.0);
    if (i == 0) { a = c0; b = c1; }
    else if (i == 1) { a = c1; b = c2; }
    else if (i == 2) { a = c2; b = c3; }
    else if (i == 3) { a = c3; b = c4; }
    else if (i == 4) { a = c4; b = c5; }
    else if (i == 5) { a = c5; b = c6; }
    else if (i == 6) { a = c6; b = c7; }
    else if (i == 7) { a = c7; b = c8; }
    else if (i == 8) { a = c8; b = c9; }
    else { a = c9; b = c10; }

    return mix(a, b, f);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 iResolution = iResolution.xy;
    vec4 iMouse = iMouse;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // 相机设置
    vec3 ro = vec3(0.0, 0.0, 2.5);
    vec3 rd = normalize(vec3(uv, -1.0));

    // 鼠标控制视角
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

            // Lighting
            vec3 normal = estimateNormal(p);
            vec3 lightDir = normalize(vec3(0.6, 0.7, 0.9));
            vec3 viewDir = normalize(-rd);

            float diff = max(dot(normal, lightDir), 0.0);
            vec3 bounceDir = normalize(vec3(0.0, -1.0, 0.0));
            float bounce = max(dot(normal, bounceDir), 0.0) * 0.2;

            // Specular
            vec3 halfDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(normal, halfDir), 0.0), 96.0);

            // Rim lighting
            float rim = pow(1.0 - max(dot(normal, viewDir), 0.0), 2.0);

            // 数据映射到颜色
            float val = sin(p.x * 3.0 + p.y * 4.0 + p.z * 2.0);
            val = val * 0.5 + 0.5;
            vec3 plasma = plasmaColor(val);

            // 提高颜色对比，增强立体感
            plasma = pow(plasma, vec3(0.9)); // 拉亮暗部

            vec3 base = plasma;
            vec3 finalColor =
                  base * diff               // 主光照
                + base * 0.3               // 环境提亮
                + base * bounce            // 地面反弹
                + vec3(1.0) * spec * 0.7   // 高光
                + base * rim * 0.3;        // 轮廓提亮

            col = finalColor;
            break;
        }
        if (t > 100.0) break;
        t += d;
    }

    vec3 bg = vec3(1.02, 1.04, 1.05); // 冷白背景
    fragColor = vec4(hit ? col : bg, 1.0);
}