float torusSDF(vec3 p) {
    //                vec2 q = vec2(length(p.xz) - 0.7, p.y);
    //                float torus = length(q) - 0.2;
    //                vec3 b = vec3(0.2);
    //                float box = length(max(abs(p - vec3(0.5, 0.0, 0.0)) - b, 0.0));
    //                return min(torus, box);
    return max(length(vec2(dot(p, vec3(-0.000000, 0.000000, 1.000000)), dot(p, vec3(-0.707107, 0.707107, -0.000000))) - vec2(0.0, 0.0)) - 0.5, abs(dot(p, vec3(-0.707107, -0.707107, 0.000000))) - 0.707107);

}