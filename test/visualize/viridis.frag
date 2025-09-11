// ---------- Constants ----------
const int   MAX_STEPS = 128;
const float EPSILON   = 0.001;
const float FAR       = 100.0;

// ---------- 你可以替换这个函数 ----------
float computeField(vec3 p) {
    // 任意可视化函数，比如：
    return sin(p.x * 3.0 + p.y * 2.0 + p.z * 4.0);
}


// ---------- SDF ----------
float torusSDF(vec3 p) {
    // Torus
    vec2 q = vec2(length(p.xz) - 0.7, p.y);
    float torus = length(q) - 0.2;

    // Box
    vec3 b = vec3(0.2);
    float box = length(max(abs(p - vec3(0.5, 0.0, 0.0)) - b, 0.0));

    return min(torus, box);
}

float sceneSDF(vec3 p) {
    return torusSDF(p);
}

// ---------- Normal ----------
vec3 estimateNormal(vec3 p) {
    float e = EPSILON;
    return normalize(vec3(
    sceneSDF(p + vec3(e, 0, 0)) - sceneSDF(p - vec3(e, 0, 0)),
    sceneSDF(p + vec3(0, e, 0)) - sceneSDF(p - vec3(0, e, 0)),
    sceneSDF(p + vec3(0, 0, e)) - sceneSDF(p - vec3(0, 0, e))
    ));
}

// ---------- Color Map ----------
vec3 viridisColor(float x) {
    x = clamp(x, 0.0, 1.0);
    // 其中数值是参考官方 viridis 插值表
    const vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    const vec3 c1 = vec3(0.190631, 0.407061, 0.556089);
    const vec3 c2 = vec3(0.20803, 0.718701, 0.472873);
    const vec3 c3 = vec3(0.993248, 0.906157, 0.143936);
    if (x < 0.33){
        float f = x/0.33;
        return mix(c0, c1, f);
    } else if (x < 0.66){
        float f = (x-0.33)/0.33;
        return mix(c1, c2, f);
    } else {
        float f = (x-0.66)/0.34;
        return mix(c2, c3, f);
    }
}

// ---------- Estimate min/max over a region ----------
void estimateMinMax(out float minVal, out float maxVal) {
    minVal =  10000.0;
    maxVal = -10000.0;

    // Sample 5x5x5 = 125 points in cube [-1,1]^3
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

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 iResolution = iResolution.xy;
    vec4 iMouse = iMouse;
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;

    // Camera
    vec3 ro = vec3(0.0, 0.0, 2.5);
    vec3 rd = normalize(vec3(uv, -1.0));

    // View rotation
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

    // Estimate min/max for current frame
    float minVal, maxVal;
    estimateMinMax(minVal, maxVal);
    float range = maxVal - minVal + 1e-5;

    // Raymarch
    float t = 0.0;
    vec3 col = vec3(0.0);
    vec3 bg = vec3(1.0);// bright background
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
            val = clamp((val - minVal) / range, 0.0, 1.0);// <--- ✅ 动态归一化
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
            + vec3(1.0) * spec * 0.1
            * tint;

            break;
        }
        if (t > FAR) break;
        t += d;
    }

    fragColor = vec4(hit ? col : bg, 1.0);
}