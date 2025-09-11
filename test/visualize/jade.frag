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

// Jet colormap: v in [0, 1]
vec3 jetColor(float v) {
    v = clamp(v, 0.0, 1.0);
    return clamp(vec3(
        1.5 - abs(4.0 * v - 3.0),
        1.5 - abs(4.0 * v - 2.0),
        1.5 - abs(4.0 * v - 1.0)
    ), 0.0, 1.0);
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
            float bounce = max(dot(normal, bounceDir), 0.0) * 0.3;

            // Specular
            vec3 halfDir = normalize(lightDir + viewDir);
            float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

            // 数据颜色（可替换为真实变量）
            float val = sin(p.x * 3.0 + p.y * 4.0 + p.z * 2.0);
            val = val * 0.5 + 0.5;
            vec3 jet = jetColor(val);

            // 玉石色调混合
            vec3 base = mix(jet, vec3(1.0), 0.2);

            // 最终颜色（包含漫反射、环境光、底部反射、高光）
            col = base * diff          // 主光源
                + base * 0.4           // 环境光（提亮）
                + base * bounce        // 底部填光
                + vec3(1.0) * spec * 0.7;  // 镜面高光

            break;
        }
        if (t > 100.0) break;
        t += d;
    }

    vec3 bg = vec3(1.0); // 背景更亮
    fragColor = vec4(hit ? col : bg, 1.0);
}