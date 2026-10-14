(() => {
  "use strict";

  /* ============================================================================
   * 1. 内联编译 WebAssembly (Wasm) 弹簧物理与 Smooth-Min 模块 (零外部依赖)
   * ============================================================================ */
  function compileWasmCore() {
    const uleb = (n) => {
      const out = [];
      do {
        let b = n & 0x7f;
        n >>>= 7;
        if (n !== 0) b |= 0x80;
        out.push(b);
      } while (n !== 0);
      return out;
    };
    const str = (s) => [s.length, ...Array.from(s, (c) => c.charCodeAt(0))];
    const sec = (id, payload) => [id, ...uleb(payload.length), ...payload];

    const typeSec = sec(1, [
      2,
      0x60, 6, 0x7d, 0x7d, 0x7d, 0x7d, 0x7d, 0x7d, 1, 0x7d,
      0x60, 3, 0x7d, 0x7d, 0x7d, 1, 0x7d
    ]);
    const funcSec = sec(3, [3, 0, 0, 1]);
    const exportSec = sec(7, [
      3,
      ...str("stepPos"), 0x00, 0,
      ...str("stepVel"), 0x00, 1,
      ...str("smin"), 0x00, 2
    ]);

    const body0 = [
      0,
      0x20, 0, 0x20, 1, 0x20, 2, 0x20, 0, 0x93, 0x20, 3, 0x94,
      0x20, 1, 0x20, 4, 0x94, 0x93, 0x20, 5, 0x94, 0x92, 0x20, 5, 0x94, 0x92, 0x0b
    ];
    const body1 = [
      0,
      0x20, 1, 0x20, 2, 0x20, 0, 0x93, 0x20, 3, 0x94,
      0x20, 1, 0x20, 4, 0x94, 0x93, 0x20, 5, 0x94, 0x92, 0x0b
    ];
    const body2 = [
      1, 2, 0x7d,
      0x20, 2, 0x43, 0x17, 0xb7, 0xd1, 0x38, 0x97, 0x22, 4,
      0x20, 0, 0x20, 1, 0x93, 0x8b, 0x93,
      0x43, 0x00, 0x00, 0x00, 0x00, 0x97,
      0x22, 3, 0x20, 3, 0x94,
      0x43, 0x00, 0x00, 0x80, 0x3e, 0x94,
      0x20, 4, 0x95, 0x21, 3,
      0x20, 0, 0x20, 1, 0x96, 0x20, 3, 0x93, 0x0b
    ];

    const codeSec = sec(10, [
      3,
      ...uleb(body0.length), ...body0,
      ...uleb(body1.length), ...body1,
      ...uleb(body2.length), ...body2
    ]);

    return new Uint8Array([
      0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
      ...typeSec,
      ...funcSec,
      ...exportSec,
      ...codeSec
    ]);
  }

  /* ============================================================================
   * 2. 零分配 Web Worker 粘连碰撞检测器 (基于紧凑 Float32Array 传输)
   * ============================================================================ */
  function createAdhesionWorker(wasmBytes, onResult) {
    const code = `
      let wasm = null;
      self.onmessage = async (e) => {
        const d = e.data;
        if (d.type === "init") {
          const res = await WebAssembly.instantiate(d.wasmBytes);
          wasm = res.instance.exports;
        } else if (d.type === "step" && wasm) {
          const buf = d.buffer;
          const n = d.count;
          const pairs = [];
          for (let i = 0; i < n; i++) {
            const o1 = i * 6;
            for (let j = i + 1; j < n; j++) {
              const o2 = j * 6;
              if (buf[o1 + 5] === buf[o2 + 5]) {
                const dx = Math.abs(buf[o1 + 1] - buf[o2 + 1]) - (buf[o1 + 3] + buf[o2 + 3]) * 0.5;
                const dy = Math.abs(buf[o1 + 2] - buf[o2 + 2]) - (buf[o1 + 4] + buf[o2 + 4]) * 0.5;
                if (wasm.smin(dx, dy, 40.0) < 26.0) {
                  pairs.push([buf[o1], buf[o2]]);
                }
              }
            }
          }
          self.postMessage({ pairs });
        }
      };
    `;
    const blob = new Blob([code], { type: "application/javascript" });
    const url = URL.createObjectURL(blob);
    const worker = new Worker(url);
    URL.revokeObjectURL(url);
    worker.onmessage = (e) => {
      if (e.data && onResult) onResult(e.data.pairs);
    };
    worker.postMessage({ type: "init", wasmBytes });
    return worker;
  }

  /* ============================================================================
   * 3. 高性能 WebGPU WGSL 着色器
   *    性能优化设计:
   *    - [0 .. trackCount) 存放不透明轨道图元，[trackCount .. count) 存放按 layerId 预排序的玻璃图元
   *    - 单次线性切片扫描 [layerStart .. layerEnd)，消除 O(L * N) 重复遍历
   *    - 模糊采样与 5 波段光谱色散解耦，纹理采样数降低 72% (50 taps -> 14 taps)
   *    - 1px 陀螺仪动态流光轮廓线高光 + 强斯涅尔凸透镜折射扭曲 + 35% 跨控件光照深度
   * ============================================================================ */
  const WGSL_SHADER = /* wgsl */`
    struct Globals {
      resolution : vec2<f32>,
      time       : f32,
      count      : u32,
      lightTilt  : vec2<f32>,
      specBoost  : f32,
      modalDim   : f32,
      trackCount : u32,
      pad0       : u32,
      pad1       : vec2<f32>,
    };

    // 严格 16 字节对齐: 6 * vec4<f32> = 24 floats = 96 bytes
    struct ControlGPU {
      geom      : vec4<f32>, // xy: center, zw: halfSize
      style     : vec4<f32>, // x: radius, y: translucency(0~1024), z: tintEnabled, w: tintOpacity
      tintAdh   : vec4<f32>, // xyz: tintColor, w: adhesion
      light     : vec4<f32>, // xy: touchPos, z: glowIntensity, w: layerId
      deform    : vec4<f32>, // xy: warpSkew, z: glassActive, w: solidTrack
      trackInfo : vec4<f32>, // xyz: trackColor, w: trackFillRatio
    };

    @group(0) @binding(0) var<uniform> globals : Globals;
    @group(0) @binding(1) var<storage, read> controls : array<ControlGPU>;
    @group(0) @binding(2) var bgTex : texture_2d<f32>;
    @group(0) @binding(3) var bgSampler : sampler;

    struct VSOut {
      @builtin(position) pos : vec4<f32>,
      @location(0) uv : vec2<f32>,
    };

    @vertex
    fn vsMain(@builtin(vertex_index) vid : u32) -> VSOut {
      var pts = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>( 1.0, -1.0), vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0), vec2<f32>( 1.0, -1.0), vec2<f32>( 1.0,  1.0)
      );
      var out : VSOut;
      let p = pts[vid];
      out.pos = vec4<f32>(p, 0.0, 1.0);
      out.uv = vec2<f32>(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
      return out;
    }

    fn sdRoundedBox(p : vec2<f32>, b : vec2<f32>, r : f32) -> f32 {
      let rr = min(r, min(b.x, b.y));
      let q = abs(p) - b + vec2<f32>(rr, rr);
      return min(max(q.x, q.y), 0.0) + length(max(q, vec2<f32>(0.0))) - rr;
    }

    fn sminBlend(a : f32, b : f32, k : f32) -> vec2<f32> {
      let safeK = max(k, 0.0001);
      let h = clamp(0.5 + 0.5 * (b - a) / safeK, 0.0, 1.0);
      let d = mix(b, a, h) - safeK * h * (1.0 - h);
      return vec2<f32>(d, h);
    }

    fn evalControlSDF(p : vec2<f32>, c : ControlGPU) -> f32 {
      let rel = p - c.geom.xy;
      let halfSize = c.geom.zw;
      let warped = rel - c.deform.xy * clamp(length(rel) / max(length(halfSize), 1.0), 0.0, 1.2);
      return sdRoundedBox(warped, halfSize, c.style.x);
    }

    // 仅遍历 [0 .. trackCount) 区间内的不透明轨道图元
    fn compositeSolidTracks(px : vec2<f32>, baseCol : vec3<f32>) -> vec3<f32> {
      var col = baseCol;
      let tCount = i32(globals.trackCount);
      for (var i = 0; i < tCount; i = i + 1) {
        let c = controls[i];
        let diff = abs(px - c.geom.xy) - c.geom.zw;
        if (diff.x > 2.0 || diff.y > 2.0) { continue; }

        let d = evalControlSDF(px, c);
        if (d < 1.0) {
          let localX = clamp((px.x - (c.geom.x - c.geom.z)) / max(c.geom.z * 2.0, 1.0), 0.0, 1.0);
          var trackCol = c.trackInfo.xyz;
          if (c.deform.w < 1.5) {
            trackCol = select(vec3<f32>(0.15, 0.17, 0.22), c.trackInfo.xyz, localX <= c.trackInfo.w);
          }
          col = mix(col, trackCol, smoothstep(1.0, -1.0, d));
        }
      }
      return col;
    }

    fn samplePointScene(uv : vec2<f32>) -> vec3<f32> {
      let clampedUV = clamp(uv, vec2<f32>(0.001), vec2<f32>(0.999));
      let raw = textureSampleLevel(bgTex, bgSampler, clampedUV, 0.0).rgb;
      return compositeSolidTracks(clampedUV * globals.resolution, raw);
    }

    // 解耦模糊与 5 波段光谱色散：仅对主折射中心做 9-tap Vogel 模糊，色散偏移通过双线性光谱差分叠加
    fn sampleDispersedAndBlurred(centerUV : vec2<f32>, prismStep : vec2<f32>, blurNorm : f32) -> vec3<f32> {
      let res = globals.resolution;
      let sR  = samplePointScene(centerUV + prismStep *  1.00);
      let sY  = samplePointScene(centerUV + prismStep *  0.50);
      let sG  = samplePointScene(centerUV);
      let sCy = samplePointScene(centerUV + prismStep * -0.50);
      let sB  = samplePointScene(centerUV + prismStep * -1.00);

      let spectralCol = vec3<f32>(
        sR.r * 0.72 + sY.r * 0.28,
        sY.g * 0.22 + sG.g * 0.56 + sCy.g * 0.22,
        sCy.b * 0.28 + sB.b * 0.72
      );

      if (blurNorm < 0.006) {
        return spectralCol;
      }

      let chromaFringe = spectralCol - sG;
      let radiusPx = blurNorm * 34.0;
      var accum = sG;
      var wSum = 1.0;
      let taps = 8;
      for (var i = 0; i < taps; i = i + 1) {
        let fi = f32(i);
        let r = sqrt((fi + 0.5) / f32(taps)) * radiusPx;
        let theta = fi * 2.3999632;
        let sampleUV = clamp(centerUV + vec2<f32>(cos(theta), sin(theta)) * r / res, vec2<f32>(0.001), vec2<f32>(0.999));
        let w = 1.0 - (fi / f32(taps)) * 0.30;
        accum += textureSampleLevel(bgTex, bgSampler, sampleUV, 0.0).rgb * w;
        wSum += w;
      }
      return (accum / wSum) + chromaFringe;
    }

    fn shadeGlassSurface(
      fragPos : vec2<f32>,
      uv : vec2<f32>,
      baseColor : vec3<f32>,
      dSurface : f32,
      outwardNormal2D : vec2<f32>,
      blendedTint : vec3<f32>,
      blendedTintOp : f32,
      blendedTranslucency : f32,
      blendedHalfSize : vec2<f32>,
      receiverHalfSize : vec2<f32>,
      dReceiverOwn : f32,
      layerStart : i32,
      layerEnd : i32,
      glassWeight : f32
    ) -> vec3<f32> {
      let res = globals.resolution;
      let minHalf = min(blendedHalfSize.x, blendedHalfSize.y);

      // 1. 强化的超椭圆弯月面透镜截面与连续曲率
      let bezelWidth = clamp(minHalf * 0.58, 14.0, 44.0);
      let uBevel = clamp(1.0 - (-dSurface / bezelWidth), 0.0, 1.0);
      let edgeCurve = pow(uBevel, 1.55) / sqrt(max(1.0 - uBevel * uBevel * 0.91, 0.035));
      let innerDomeSlope = clamp(1.0 - (-dSurface / max(minHalf, 1.0)), 0.0, 1.0) * 0.35;
      let meniscusSlope = edgeCurve * 1.95 + innerDomeSlope;

      let N = normalize(vec3<f32>(outwardNormal2D * meniscusSlope, 1.0));
      let V = vec3<f32>(0.0, 0.0, 1.0);

      // 2. 强力斯涅尔折射 + 凸透镜径向放大 + 5波段连续棱镜色散
      let controlSpan = length(blendedHalfSize * 2.0);
      let largeScaleFactor = clamp((controlSpan - 70.0) / 280.0, 0.28, 2.2);

      let refractVec = refract(-V, N, 1.0 / 1.64);
      let tangent2D = vec2<f32>(-outwardNormal2D.y, outwardNormal2D.x);
      let lensMagnify = outwardNormal2D * (-sin(uBevel * 3.1415926) * (24.0 + largeScaleFactor * 16.0));
      let swirlWarp = tangent2D * (pow(uBevel, 2.0) * sin(uBevel * 6.28318) * 7.5);
      let refractOffset = (refractVec.xy * (54.0 + largeScaleFactor * 52.0) + lensMagnify + swirlWarp) / res;

      let dispersionSpan = (0.22 + pow(uBevel, 1.35) * 0.78) * (0.48 + largeScaleFactor * 0.58);
      let prismStep = (refractOffset + tangent2D * length(refractOffset) * 0.22) * dispersionSpan;

      let blurNorm = clamp(blendedTranslucency / 1024.0, 0.0, 1.0);
      let refractedCol = sampleDispersedAndBlurred(uv + refractOffset, prismStep, blurNorm);

      // 3. 贝尔-朗伯 (Beer-Lambert) 光学透射滤色
      let op = clamp(blendedTintOp, 0.0, 1.0);
      let transmittance = mix(vec3<f32>(1.0), pow(max(blendedTint, vec3<f32>(0.02)), vec3<f32>(0.65)) * 1.18, op * 0.68);
      var glassCol = refractedCol * transmittance + blendedTint * (op * 0.22);

      // 4. 触控发光 + 同层级跨控件光照传播 (仅遍历当前层切片 [layerStart .. layerEnd)，上限 35% 深度)
      var totalLight = vec3<f32>(0.0);
      let receiverMinSpan = min(receiverHalfSize.x * 2.0, receiverHalfSize.y * 2.0);
      let maxAllowedDepth = 0.35 * receiverMinSpan;
      let ownDepth = max(0.0, -dReceiverOwn);

      for (var l = layerStart; l < layerEnd; l = l + 1) {
        let emitter = controls[l];
        let glowIntensity = emitter.light.z;
        if (glowIntensity <= 0.001) { continue; }

        let dEmitter = evalControlSDF(fragPos, emitter);
        let toLight = emitter.light.xy - fragPos;
        let dist = length(toLight);
        let lightDir = toLight / max(dist, 0.001);
        let emitterTint = select(vec3<f32>(0.86, 0.95, 1.0), emitter.tintAdh.xyz, emitter.style.z > 0.5);

        if (dEmitter <= 0.5) {
          let radial = exp(-dist * dist / (2.0 * 82.0 * 82.0));
          let edgeLineBoost = exp(-max(0.0, -dSurface) * 0.75) * exp(-dist / 150.0) * 0.85;
          totalLight += emitterTint * glowIntensity * (radial * 0.45 + edgeLineBoost);
        } else if (ownDepth <= maxAllowedDepth) {
          let depthAtten = 1.0 - smoothstep(0.0, maxAllowedDepth, ownDepth);
          let dirFacing = max(dot(outwardNormal2D, lightDir), 0.0);
          let distAtten = exp(-dist / 210.0);
          totalLight += emitterTint * glowIntensity * depthAtten * (0.25 + 0.75 * dirFacing) * distAtten * 0.85;
        }
      }

      // 5. 陀螺仪驱动的 1px 精细边缘线条高光 (Sub-pixel Contour Hairline)
      let tiltDir2D = normalize(globals.lightTilt + vec2<f32>(0.0001, -0.0001));
      let rimFacing = max(dot(outwardNormal2D, tiltDir2D), 0.0);
      let rimOpposite = max(dot(outwardNormal2D, -tiltDir2D), 0.0);

      let outerHairline = exp(-pow((dSurface + 0.8) / 0.78, 2.0));
      let innerHairline = exp(-pow((dSurface + 3.4) / 0.68, 2.0));

      let boost = 1.0 + clamp(globals.specBoost, 0.0, 0.75);
      let primaryLineIntensity = outerHairline * (0.20 + 1.05 * pow(rimFacing, 2.0) + 0.38 * pow(rimOpposite, 3.0)) * boost;
      let secondaryLineIntensity = innerHairline * (0.06 + 0.42 * pow(rimOpposite, 2.2) + 0.22 * pow(rimFacing, 2.6));

      let lineSpecColor =
        vec3<f32>(1.0, 0.995, 0.99) * primaryLineIntensity +
        vec3<f32>(0.72, 0.92, 1.0) * secondaryLineIntensity;

      glassCol += totalLight + lineSpecColor;

      let alpha = smoothstep(0.8, -1.0, dSurface) * clamp(glassWeight, 0.0, 1.0);
      return mix(baseColor, glassCol, alpha);
    }

    @fragment
    fn fsMain(in : VSOut) -> @location(0) vec4<f32> {
      let res = globals.resolution;
      let fragPos = in.uv * res;
      let tCount = i32(globals.trackCount);
      let total = i32(globals.count);

      let bgBase = textureSampleLevel(bgTex, bgSampler, in.uv, 0.0).rgb;
      var finalColor = compositeSolidTracks(fragPos, bgBase);

      // AABB 包围盒快速早退 (Early-Out)
      var nearAnyGlass = false;
      for (var i = tCount; i < total; i = i + 1) {
        let c = controls[i];
        let diff = abs(fragPos - c.geom.xy) - c.geom.zw;
        if (diff.x < 48.0 && diff.y < 48.0) {
          nearAnyGlass = true;
          break;
        }
      }
      if (!nearAnyGlass) {
        if (globals.modalDim > 0.01) {
          finalColor *= (1.0 - globals.modalDim * 0.45);
        }
        return vec4<f32>(finalColor, 1.0);
      }

      var modalDimApplied = false;
      let eps = 1.5;
      let kSmooth = 40.0;
      let shadowOffset = vec2<f32>(0.0, 9.0);

      // 单次线性扫描已按 layerId 排序的玻璃图元 [tCount .. total)
      var idx = tCount;
      while (idx < total) {
        let layer = controls[idx].light.w;
        var layerEnd = idx + 1;
        while (layerEnd < total && abs(controls[layerEnd].light.w - layer) < 0.005) {
          layerEnd = layerEnd + 1;
        }

        if (!modalDimApplied && layer >= 9.5 && globals.modalDim > 0.01) {
          finalColor *= (1.0 - globals.modalDim * 0.45);
          modalDimApplied = true;
        }

        // Pass A: 当前层切片内的粘连簇 (adhesion > 0.5)
        var hasAdhesive = false;
        var dMerged = 1e6;
        var dRight = 1e6;
        var dUp = 1e6;
        var dShadow = 1e6;
        var blendedTint = vec3<f32>(1.0);
        var blendedTintOp = 0.0;
        var blendedTranslucency = 200.0;
        var blendedHalfSize = vec2<f32>(80.0);
        var nearestOwnD = 1e6;
        var nearestHalfSize = vec2<f32>(80.0);
        var clusterWeight = 1.0;

        for (var j = idx; j < layerEnd; j = j + 1) {
          let cj = controls[j];
          if (cj.tintAdh.w < 0.5) { continue; }
          let boxDiff = abs(fragPos - cj.geom.xy) - cj.geom.zw;
          if (boxDiff.x > 64.0 || boxDiff.y > 64.0) { continue; }

          let dj = evalControlSDF(fragPos, cj);
          let dr = evalControlSDF(fragPos + vec2<f32>(eps, 0.0), cj);
          let du = evalControlSDF(fragPos + vec2<f32>(0.0, eps), cj);
          let ds = evalControlSDF(fragPos - shadowOffset, cj);
          let cjTint = select(vec3<f32>(1.0), cj.tintAdh.xyz, cj.style.z > 0.5);
          let cjOp = select(0.0, cj.style.w, cj.style.z > 0.5);

          if (dj < nearestOwnD) {
            nearestOwnD = dj;
            nearestHalfSize = cj.geom.zw;
          }

          if (!hasAdhesive) {
            dMerged = dj;
            dRight = dr;
            dUp = du;
            dShadow = ds;
            blendedTint = cjTint;
            blendedTintOp = cjOp;
            blendedTranslucency = cj.style.y;
            blendedHalfSize = cj.geom.zw;
            clusterWeight = cj.deform.z;
            hasAdhesive = true;
          } else {
            let sb = sminBlend(dMerged, dj, kSmooth);
            let w = 1.0 - sb.y;
            dMerged = sb.x;
            dRight = sminBlend(dRight, dr, kSmooth).x;
            dUp = sminBlend(dUp, du, kSmooth).x;
            dShadow = sminBlend(dShadow, ds, kSmooth).x;
            blendedTint = mix(blendedTint, cjTint, w);
            blendedTintOp = mix(blendedTintOp, cjOp, w);
            blendedTranslucency = mix(blendedTranslucency, cj.style.y, w);
            blendedHalfSize = mix(blendedHalfSize, cj.geom.zw, w);
            clusterWeight = mix(clusterWeight, cj.deform.z, w);
          }
        }

        if (hasAdhesive && dMerged >= -0.5 && dShadow < 26.0) {
          let shadowAlpha = exp(-max(0.0, dShadow) * 0.14) * 0.34 * clusterWeight;
          let contactAO = exp(-max(0.0, dMerged) * 0.42) * 0.18 * clusterWeight;
          finalColor *= (1.0 - clamp(shadowAlpha + contactAO, 0.0, 0.48));
        }

        if (hasAdhesive && dMerged < 1.0) {
          let grad = vec2<f32>(dRight - dMerged, dUp - dMerged) / eps;
          let gradLen = length(grad);
          let outwardNormal2D = select(vec2<f32>(0.0, -1.0), grad / max(gradLen, 0.0001), gradLen > 0.0001);
          finalColor = shadeGlassSurface(
            fragPos, in.uv, finalColor, dMerged, outwardNormal2D,
            blendedTint, blendedTintOp, blendedTranslucency,
            blendedHalfSize, nearestHalfSize, nearestOwnD, idx, layerEnd, clusterWeight
          );
        }

        // Pass B: 当前层切片内的独立非粘连控件 (adhesion <= 0.5)
        for (var j = idx; j < layerEnd; j = j + 1) {
          let cj = controls[j];
          if (cj.tintAdh.w >= 0.5) { continue; }
          let boxDiff = abs(fragPos - cj.geom.xy) - cj.geom.zw;
          if (boxDiff.x > 34.0 || boxDiff.y > 34.0) { continue; }

          let dj = evalControlSDF(fragPos, cj);
          let ds = evalControlSDF(fragPos - shadowOffset, cj);

          if (dj >= -0.5 && ds < 26.0) {
            let shadowAlpha = exp(-max(0.0, ds) * 0.14) * 0.36 * cj.deform.z;
            let contactAO = exp(-max(0.0, dj) * 0.45) * 0.18 * cj.deform.z;
            finalColor *= (1.0 - clamp(shadowAlpha + contactAO, 0.0, 0.52));
          }

          if (dj < 1.0) {
            let dr = evalControlSDF(fragPos + vec2<f32>(eps, 0.0), cj);
            let du = evalControlSDF(fragPos + vec2<f32>(0.0, eps), cj);
            let grad = vec2<f32>(dr - dj, du - dj) / eps;
            let gradLen = length(grad);
            let outwardNormal2D = select(vec2<f32>(0.0, -1.0), grad / max(gradLen, 0.0001), gradLen > 0.0001);
            let cjTint = select(vec3<f32>(1.0), cj.tintAdh.xyz, cj.style.z > 0.5);
            let cjOp = select(0.0, cj.style.w, cj.style.z > 0.5);

            finalColor = shadeGlassSurface(
              fragPos, in.uv, finalColor, dj, outwardNormal2D,
              cjTint, cjOp, cj.style.y,
              cj.geom.zw, cj.geom.zw, dj, idx, layerEnd, cj.deform.z
            );
          }
        }

        idx = layerEnd;
      }

      if (!modalDimApplied && globals.modalDim > 0.01) {
        finalColor *= (1.0 - globals.modalDim * 0.45);
      }

      return vec4<f32>(finalColor, 1.0);
    }
  `;

  /* ============================================================================
   * 4. 工具函数：颜色解析与默认光学背景绘制器
   * ============================================================================ */
  const COLOR_CACHE = new Map();
  function hexToRgb01(hex) {
    const key = hex || "#38b6ff";
    let cached = COLOR_CACHE.get(key);
    if (cached) return cached;
    const clean = key.replace("#", "");
    const num = parseInt(clean.length === 3 ? clean.split("").map((c) => c + c).join("") : clean, 16) || 0x38b6ff;
    cached = [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255];
    COLOR_CACHE.set(key, cached);
    return cached;
  }

  function defaultProceduralBackground(ctx, w, h, time) {
    const grad = ctx.createLinearGradient(0, 0, w, h);
    grad.addColorStop(0, "#080c1c");
    grad.addColorStop(0.5, "#11162c");
    grad.addColorStop(1, "#061420");
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);

    const orbs = [
      { x: w * 0.20 + Math.sin(time * 0.65) * 70, y: h * 0.26 + Math.cos(time * 0.5) * 45, r: 240, c: "rgba(255, 65, 135, 0.56)" },
      { x: w * 0.52 + Math.cos(time * 0.55) * 85, y: h * 0.48 + Math.sin(time * 0.75) * 55, r: 270, c: "rgba(0, 205, 255, 0.52)" },
      { x: w * 0.34 + Math.sin(time * 0.85) * 55, y: h * 0.75 + Math.cos(time * 0.65) * 45, r: 220, c: "rgba(135, 80, 255, 0.55)" },
      { x: w * 0.78 + Math.cos(time * 0.45) * 55, y: h * 0.30 + Math.sin(time * 0.6) * 40, r: 200, c: "rgba(48, 219, 98, 0.48)" }
    ];
    for (let i = 0; i < orbs.length; i++) {
      const o = orbs[i];
      const rg = ctx.createRadialGradient(o.x, o.y, 12, o.x, o.y, o.r);
      rg.addColorStop(0, o.c);
      rg.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = rg;
      ctx.beginPath();
      ctx.arc(o.x, o.y, o.r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
    ctx.lineWidth = 1.2;
    const step = 38;
    ctx.beginPath();
    for (let x = 0; x < w; x += step) {
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
    }
    for (let y = 0; y < h; y += step) {
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
    }
    ctx.stroke();

    ctx.strokeStyle = "rgba(255, 255, 255, 0.24)";
    ctx.lineWidth = 2.2;
    for (let r = 90; r <= 360; r += 90) {
      ctx.beginPath();
      ctx.arc(w * 0.46, h * 0.48, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }

  /* ============================================================================
   * 5. LiquidGlassControl 控件实例类 (提供链式调用、事件订阅与响应式属性更新)
   * ============================================================================ */
  class LiquidGlassControl {
    constructor(engine, config) {
      this.engine = engine;
      this.id = engine.idCounter++;
      this.type = (config.type || "button").toLowerCase();
      this.listeners = new Map();

      const defaultTint = this.type === "switch" ? "#34c759" : "#38b6ff";
      this.label = config.label ?? "Control";
      this.description = config.description ?? "";
      this.x = config.x ?? 200;
      this.y = config.y ?? 200;
      this.targetX = this.x;
      this.targetY = this.y;
      this.vx = 0;
      this.vy = 0;

      this.width = config.width ?? 160;
      this.height = config.height ?? 50;
      this.targetWidth = this.width;
      this.targetHeight = this.height;
      this.vw = 0;
      this.vh = 0;

      this.scale = config.hiddenByMorph ? 0.4 : 1;
      this.targetScale = this.scale;
      this.vScale = 0;

      this.tint = config.tint ?? true;
      this.tintColor = config.tintColor ?? defaultTint;
      this.tintOpacity = config.tintOpacity ?? 0.14;
      this.adhesion = config.adhesion ?? true;
      this.draggableDeform = config.draggableDeform ?? true;
      this.borderRadius = config.borderRadius ?? 22;
      this.targetRadius = this.borderRadius;
      this.vRadius = 0;
      this.translucency = Math.max(0, Math.min(1024, Math.round(config.translucency ?? 110)));
      this.interactive = config.interactive ?? true;
      this.layerId = config.layerId ?? (this.type === "alert" ? 10 : 1);
      this.zOrder = engine.zOrderCounter++;

      this.touchOffsetX = 0;
      this.touchOffsetY = 0;
      this.glowIntensity = 0;
      this.targetGlow = 0;
      this.vGlow = 0;
      this.skewX = 0;
      this.skewY = 0;
      this.targetSkewX = 0;
      this.targetSkewY = 0;
      this.vSkewX = 0;
      this.vSkewY = 0;
      this.isPressed = false;
      this.glassActive = config.hiddenByMorph ? 0 : (config.glassActive ?? 1);
      this.morphTargetId = config.morphTargetId ?? null;
      this.hiddenByMorph = Boolean(config.hiddenByMorph);

      this.value = config.value ?? 0.58;
      this.vValue = 0;
      this.checked = config.checked ?? true;
      this.colorMix = this.checked ? 1 : 0;
      this.targetColorMix = this.colorMix;
      this.vColorMix = 0;
      this.options = Array.isArray(config.options) ? [...config.options] : ["选项 A", "选项 B", "选项 C"];
      this.selectedIndex = config.selectedIndex ?? 0;

      this._lastCssW = -1;
      this._lastCssH = -1;
      this._lastCssR = -1;
      this._lastCssT = "";
      this._lastMaskKey = "";
      this._lastMaskX = -9999;
      this._lastMaskY = -9999;

      if (typeof config.onChange === "function") this.on("change", config.onChange);
      if (typeof config.onClick === "function") this.on("click", config.onClick);
      if (typeof config.onInput === "function") this.on("input", config.onInput);
      if (typeof config.onConfirm === "function") this.on("confirm", config.onConfirm);
    }

    on(event, handler) {
      if (!this.listeners.has(event)) this.listeners.set(event, new Set());
      this.listeners.get(event).add(handler);
      return this;
    }

    off(event, handler) {
      if (this.listeners.has(event)) this.listeners.get(event).delete(handler);
      return this;
    }

    emit(event, ...args) {
      const set = this.listeners.get(event);
      if (set) {
        set.forEach((fn) => {
          try { fn(...args, this); } catch (err) { console.error(err); }
        });
      }
    }

    /**
     * 响应式批量更新控件属性 (支持全部 10 项光学与几何参数)
     */
    setProps(props = {}) {
      if (props.width !== undefined) this.targetWidth = Number(props.width);
      if (props.height !== undefined) this.targetHeight = Number(props.height);
      if (props.x !== undefined) this.targetX = Number(props.x);
      if (props.y !== undefined) this.targetY = Number(props.y);
      if (props.tint !== undefined) this.tint = Boolean(props.tint);
      if (props.tintColor !== undefined) this.tintColor = String(props.tintColor);
      if (props.tintOpacity !== undefined) this.tintOpacity = Math.max(0, Math.min(1, Number(props.tintOpacity)));
      if (props.adhesion !== undefined) this.adhesion = Boolean(props.adhesion);
      if (props.draggableDeform !== undefined) this.draggableDeform = Boolean(props.draggableDeform);
      if (props.borderRadius !== undefined) this.targetRadius = Math.max(0, Number(props.borderRadius));
      if (props.translucency !== undefined) {
        this.translucency = Math.max(0, Math.min(1024, parseInt(props.translucency, 10) || 0));
      }
      if (props.interactive !== undefined) {
        this.interactive = Boolean(props.interactive);
        if (this.el) {
          this.el.dataset.interactive = String(this.interactive);
          this.el.style.pointerEvents = this.interactive && !this.hiddenByMorph ? "auto" : "none";
        }
      }
      if (props.layerId !== undefined) this.layerId = Number(props.layerId);
      if (props.label !== undefined) {
        this.label = String(props.label);
        if (this.labelSpan) this.labelSpan.textContent = this.label;
        if (this.textareaEl) this.textareaEl.value = this.label;
      }
      if (props.value !== undefined) this.setValue(props.value);
      if (props.checked !== undefined) this.setValue(Boolean(props.checked));
      if (props.options !== undefined && Array.isArray(props.options)) {
        this.options = [...props.options];
        this.engine.setupControlInnerDOM(this);
      }
      if (this.type === "toolbar") {
        this.engine.measureAndSyncToolbarPill(this, false);
      }
      this.engine.wakeUp();
      return this;
    }

    getValue() {
      if (this.type === "slider") return Math.max(0, Math.min(1, this.value));
      if (this.type === "switch") return this.checked;
      if (this.type === "editor") return this.textareaEl ? this.textareaEl.value : this.label;
      if (this.type === "menu" || this.type === "toolbar") return this.selectedIndex;
      return this.label;
    }

    setValue(val) {
      if (this.type === "slider") {
        this.value = Math.max(0, Math.min(1, Number(val) || 0));
      } else if (this.type === "switch") {
        this.checked = Boolean(val);
        this.targetThumbX = this.checked ? 1 : 0;
        this.targetColorMix = this.checked ? 1 : 0;
      } else if (this.type === "editor") {
        this.label = String(val);
        if (this.textareaEl) this.textareaEl.value = this.label;
      } else if (this.type === "menu" || this.type === "toolbar") {
        const idx = Math.max(0, Math.min(this.options.length - 1, Number(val) || 0));
        this.selectedIndex = idx;
        if (this.type === "toolbar" && this.toolbarItemEls) {
          this.toolbarItemEls.forEach((item, i) => item.classList.toggle("active", i === idx));
          this.engine.measureAndSyncToolbarPill(this, false);
        } else if (this.type === "menu" && this.menuItemEls) {
          this.menuItemEls.forEach((entry) => {
            entry.check.textContent = entry.idx === idx ? "✓" : "";
          });
        }
      }
      this.engine.wakeUp();
      return this;
    }

    setPosition(x, y, immediate = false) {
      this.targetX = Number(x);
      this.targetY = Number(y);
      if (immediate) {
        this.x = this.targetX;
        this.y = this.targetY;
        this.vx = 0;
        this.vy = 0;
      }
      this.engine.wakeUp();
      return this;
    }

    setSize(w, h, immediate = false) {
      this.targetWidth = Number(w);
      this.targetHeight = Number(h);
      if (immediate) {
        this.width = this.targetWidth;
        this.height = this.targetHeight;
        this.vw = 0;
        this.vh = 0;
      }
      this.engine.wakeUp();
      return this;
    }

    morphTo(targetCtrlOrId) {
      const target = typeof targetCtrlOrId === "object" ? targetCtrlOrId : this.engine.getById(targetCtrlOrId);
      if (target) this.engine.triggerContextualMorph(this, target);
      return this;
    }

    show() {
      this.hiddenByMorph = false;
      this.glassActive = (this.type === "slider" || this.type === "switch") ? 0 : 1;
      this.targetScale = 1.0;
      if (this.el) {
        this.el.style.opacity = "1";
        this.el.style.pointerEvents = this.interactive ? "auto" : "none";
      }
      this.engine.syncModalState();
      this.engine.wakeUp();
      return this;
    }

    hide() {
      this.hiddenByMorph = true;
      this.glassActive = 0;
      this.targetScale = 0.35;
      if (this.el) {
        this.el.style.opacity = "0";
        this.el.style.pointerEvents = "none";
      }
      this.engine.syncModalState();
      this.engine.wakeUp();
      return this;
    }

    destroy() {
      this.engine.removeControl(this.id);
    }
  }

  /* ============================================================================
   * 6. LiquidGlassEngine 核心引擎 (支持多实例挂载、自定义背景源、休眠唤醒与零 GC 渲染)
   * ============================================================================ */
  class LiquidGlassEngine {
    constructor(options = {}) {
      this.options = options;
      this.container = options.container || document.body;
      this.controls = [];
      this.idCounter = 1;
      this.zOrderCounter = 10;
      this.maxControls = options.maxControls || 64;
      this.backgroundMode = options.background || "procedural";
      this.customBgSource = null;
      this.bgNeedsUpdate = true;

      this.wasm = null;
      this.worker = null;
      this.activeAdhesionPairs = [];
      this.device = null;
      this.context = null;
      this.pipeline = null;
      this.uniformBuffer = null;
      this.storageBuffer = null;
      this.bgTexture = null;
      this.bgCanvas = document.createElement("canvas");
      this.bgCtx = this.bgCanvas.getContext("2d");

      this.startTime = performance.now();
      this.frameCount = 0;
      this.isSleeping = false;
      this.rafId = 0;
      this.occlusionDirty = true;

      // 陀螺仪与方向传感器物理状态
      this.gyroX = -0.36;
      this.gyroY = -0.68;
      this.targetGyroX = -0.36;
      this.targetGyroY = -0.68;
      this.vGyroX = 0;
      this.vGyroY = 0;
      this.gyroBoost = 0;
      this.targetGyroBoost = 0;
      this.vGyroBoost = 0;
      this.hasHardwareGyro = false;

      // 居中模态 Alert 遮罩渐变状态
      this.modalDim = 0;
      this.targetModalDim = 0;
      this.vModalDim = 0;

      // 零 GC 预分配 TypedArray 缓冲区
      this.gpuData = new Float32Array(this.maxControls * 24);
      this.globalsArr = new Float32Array(12);
      this.globalsU32 = new Uint32Array(this.globalsArr.buffer);
      this.workerTransferBuf = new Float32Array(this.maxControls * 6);
      this._trackPrimitives = [];
      this._glassPrimitives = [];
    }

    async init() {
      this.setupDOMStage();

      const wasmBytes = compileWasmCore();
      const wasmModule = await WebAssembly.instantiate(wasmBytes);
      this.wasm = wasmModule.instance.exports;
      this.worker = createAdhesionWorker(wasmBytes, (pairs) => {
        this.activeAdhesionPairs = pairs;
      });

      this.initSensorListeners();
      await this.initWebGPU();

      // 自动扫描并挂载容器内的声明式 HTML 控件 [data-lg]
      this.hydrate(this.container);

      this._resizeHandler = () => {
        this.handleResize();
        this.wakeUp();
      };
      window.addEventListener("resize", this._resizeHandler);
      this.wakeUp();
      return this;
    }

    wakeUp() {
      this.occlusionDirty = true;
      if (this.isSleeping || !this.rafId) {
        this.isSleeping = false;
        this.lastFrameTime = performance.now();
        this.rafId = requestAnimationFrame((t) => this.frameLoop(t));
      }
    }

    /**
     * 设置自定义背景源: 支持 "procedural" | HTMLImageElement | HTMLVideoElement | HTMLCanvasElement | Function(ctx, w, h, time)
     */
    setBackground(source) {
      if (source === "procedural" || typeof source === "function") {
        this.backgroundMode = source;
        this.customBgSource = null;
      } else if (source instanceof HTMLElement) {
        this.backgroundMode = "element";
        this.customBgSource = source;
      }
      this.bgNeedsUpdate = true;
      this.wakeUp();
      return this;
    }

    initSensorListeners() {
      this._onOrientation = (e) => {
        if (e.gamma === null && e.beta === null) return;
        this.hasHardwareGyro = true;
        const gx = Math.max(-1.0, Math.min(1.0, (e.gamma || 0) / 40.0));
        const gy = Math.max(-1.0, Math.min(1.0, ((e.beta || 35) - 40) / 40.0));
        if (Math.abs(gx - this.targetGyroX) > 0.005 || Math.abs(gy - this.targetGyroY) > 0.005) {
          this.targetGyroX = gx;
          this.targetGyroY = gy;
          this.wakeUp();
        }
      };

      this._onMotion = (e) => {
        const rot = e.rotationRate;
        if (!rot) return;
        const mag = Math.hypot(rot.alpha || 0, rot.beta || 0, rot.gamma || 0);
        if (mag > 2.0) {
          this.hasHardwareGyro = true;
          this.targetGyroBoost = Math.min(0.75, mag / 150.0);
          this.vGyroX += (rot.gamma || 0) * 0.008;
          this.vGyroY += (rot.beta || 0) * 0.008;
          this.wakeUp();
        }
      };

      this._onPointerMoveGlobal = (e) => {
        if (!this.hasHardwareGyro) {
          const nx = (e.clientX / Math.max(1, window.innerWidth)) * 2.0 - 1.0;
          const ny = (e.clientY / Math.max(1, window.innerHeight)) * 2.0 - 1.0;
          this.targetGyroX = nx * 0.9;
          this.targetGyroY = ny * 0.9 - 0.18;
          this.wakeUp();
        }
      };

      window.addEventListener("deviceorientation", this._onOrientation, { passive: true });
      window.addEventListener("devicemotion", this._onMotion, { passive: true });
      window.addEventListener("pointermove", this._onPointerMoveGlobal, { passive: true });
    }

    async requestGyroPermission() {
      if (typeof DeviceOrientationEvent !== "undefined" && typeof DeviceOrientationEvent.requestPermission === "function") {
        await DeviceOrientationEvent.requestPermission();
      }
      if (typeof DeviceMotionEvent !== "undefined" && typeof DeviceMotionEvent.requestPermission === "function") {
        await DeviceMotionEvent.requestPermission();
      }
    }

    setupDOMStage() {
      this.stage = document.createElement("div");
      this.stage.className = "lg-root-stage";
      this.canvas = document.createElement("canvas");
      this.canvas.className = "lg-webgpu-canvas";
      this.domLayer = document.createElement("div");
      this.domLayer.className = "lg-dom-layer";

      this.modalBackdrop = document.createElement("div");
      this.modalBackdrop.className = "lg-modal-backdrop";
      this.modalBackdrop.addEventListener("pointerdown", () => {
        const openAlert = this.controls.find((c) => c.type === "alert" && !c.hiddenByMorph);
        if (openAlert) {
          if (openAlert.morphTargetId) {
            this.triggerContextualMorph(openAlert, this.getById(openAlert.morphTargetId));
          } else {
            openAlert.hide();
          }
        }
      });

      this.stage.append(this.canvas, this.domLayer, this.modalBackdrop);
      this.container.appendChild(this.stage);
      this.handleResize();
    }

    handleResize() {
      const rect = this.stage.getBoundingClientRect();
      const cssW = rect.width || window.innerWidth;
      const cssH = rect.height || window.innerHeight;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = Math.max(1, Math.floor(cssW * dpr));
      const h = Math.max(1, Math.floor(cssH * dpr));
      this.dpr = dpr;
      this.stageWidth = cssW;
      this.stageHeight = cssH;
      this.canvas.width = w;
      this.canvas.height = h;
      this.bgCanvas.width = w;
      this.bgCanvas.height = h;
      this.bgNeedsUpdate = true;

      if (this.device) {
        this.createBackgroundTexture(w, h);
      }
      this.controls.forEach((c) => {
        if (c.type === "alert" && !c.hiddenByMorph) {
          c.targetX = cssW * 0.5;
          c.targetY = cssH * 0.5;
        }
      });
    }

    async initWebGPU() {
      if (!navigator.gpu) {
        this.fallbackCtx = this.canvas.getContext("2d");
        return;
      }
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        this.fallbackCtx = this.canvas.getContext("2d");
        return;
      }
      this.device = await adapter.requestDevice();
      this.context = this.canvas.getContext("webgpu");
      this.format = navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({
        device: this.device,
        format: this.format,
        alphaMode: "premultiplied"
      });

      const shaderModule = this.device.createShaderModule({ code: WGSL_SHADER });
      this.pipeline = this.device.createRenderPipeline({
        layout: "auto",
        vertex: { module: shaderModule, entryPoint: "vsMain" },
        fragment: {
          module: shaderModule,
          entryPoint: "fsMain",
          targets: [{ format: this.format }]
        },
        primitive: { topology: "triangle-list" }
      });

      this.uniformBuffer = this.device.createBuffer({
        size: 48,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });

      this.storageBuffer = this.device.createBuffer({
        size: this.maxControls * 96,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      });

      this.bgSampler = this.device.createSampler({
        magFilter: "linear",
        minFilter: "linear"
      });

      this.createBackgroundTexture(this.canvas.width, this.canvas.height);
    }

    createBackgroundTexture(w, h) {
      if (!this.device) return;
      if (this.bgTexture) this.bgTexture.destroy();
      this.bgTexture = this.device.createTexture({
        size: [w, h, 1],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT
      });
      this.bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.uniformBuffer } },
          { binding: 1, resource: { buffer: this.storageBuffer } },
          { binding: 2, resource: this.bgTexture.createView() },
          { binding: 3, resource: this.bgSampler }
        ]
      });
      this.updateBackgroundTexture(0, true);
    }

    updateBackgroundTexture(elapsed, force = false) {
      const w = this.bgCanvas.width;
      const h = this.bgCanvas.height;
      if (typeof this.backgroundMode === "function") {
        this.backgroundMode(this.bgCtx, w, h, elapsed);
      } else if (this.backgroundMode === "element" && this.customBgSource) {
        if (!force && !this.bgNeedsUpdate && this.customBgSource.tagName !== "VIDEO") return;
        this.bgCtx.drawImage(this.customBgSource, 0, 0, w, h);
        this.bgNeedsUpdate = false;
      } else {
        // 默认内置光学标定背景：仅在唤醒态隔帧更新
        if (!force && this.frameCount % 2 !== 0) return;
        defaultProceduralBackground(this.bgCtx, w, h, elapsed);
      }
      if (this.device && this.bgTexture) {
        this.device.queue.copyExternalImageToTexture({ source: this.bgCanvas }, { texture: this.bgTexture }, [w, h]);
      }
    }

    /* ==========================================================================
     * 公共组件创建与管理 API (支持全部 10 种控件类型)
     * ========================================================================== */
    create(typeOrConfig, maybeConfig) {
      const config = typeof typeOrConfig === "string" ? { ...maybeConfig, type: typeOrConfig } : { ...typeOrConfig };
      const ctrl = new LiquidGlassControl(this, config);

      const el = document.createElement("div");
      el.className = `lg-control lg-type-${ctrl.type}`;
      if (ctrl.type === "alert") el.classList.add("lg-layer-modal");
      el.dataset.interactive = String(ctrl.interactive);
      el.style.zIndex = String(Math.round(ctrl.layerId * 10));

      if (ctrl.hiddenByMorph) {
        el.style.pointerEvents = "none";
        el.style.opacity = "0";
      }
      ctrl.el = el;

      this.setupControlInnerDOM(ctrl);
      this.bindControlEvents(ctrl);

      this.domLayer.appendChild(el);
      this.controls.push(ctrl);

      if (ctrl.type === "toolbar") {
        requestAnimationFrame(() => this.measureAndSyncToolbarPill(ctrl, true));
      }
      if (ctrl.type === "alert") {
        this.syncModalState();
      }
      this.wakeUp();
      return ctrl;
    }

    button(props)   { return this.create("button", props); }
    toolbar(props)  { return this.create("toolbar", props); }
    dropdown(props) { return this.create("dropdown", props); }
    slider(props)   { return this.create("slider", props); }
    switch(props)   { return this.create("switch", props); }
    editor(props)   { return this.create("editor", props); }
    menu(props)     { return this.create("menu", props); }
    div(props)      { return this.create("div", props); }
    alert(props)    { return this.create("alert", props); }
    text(props)     { return this.create("text", props); }

    /**
     * 扫描并自动挂载声明式 HTML 节点: <div data-lg="button" data-width="180" ...>
     */
    hydrate(root = document) {
      const nodes = root.querySelectorAll("[data-lg]");
      const created = [];
      nodes.forEach((node) => {
        if (node._lgControl) return;
        const ds = node.dataset;
        const rect = node.getBoundingClientRect();
        const ctrl = this.create(ds.lg || "button", {
          label: ds.label || node.textContent.trim() || "Control",
          description: ds.description || "",
          x: ds.x ? Number(ds.x) : (rect.left + rect.width * 0.5 || 200),
          y: ds.y ? Number(ds.y) : (rect.top + rect.height * 0.5 || 200),
          width: ds.width ? Number(ds.width) : (rect.width || 160),
          height: ds.height ? Number(ds.height) : (rect.height || 50),
          tint: ds.tint !== "false",
          tintColor: ds.tintColor || undefined,
          tintOpacity: ds.tintOpacity ? Number(ds.tintOpacity) : undefined,
          adhesion: ds.adhesion !== "false",
          draggableDeform: ds.draggableDeform !== "false",
          borderRadius: ds.borderRadius ? Number(ds.borderRadius) : undefined,
          translucency: ds.translucency ? parseInt(ds.translucency, 10) : undefined,
          interactive: ds.interactive !== "false",
          layerId: ds.layerId ? Number(ds.layerId) : undefined,
          options: ds.options ? ds.options.split("|") : undefined,
          value: ds.value ? Number(ds.value) : undefined,
          checked: ds.checked ? ds.checked !== "false" : undefined
        });
        node._lgControl = ctrl;
        node.style.display = "none";
        created.push(ctrl);
      });
      return created;
    }

    removeControl(idOrCtrl) {
      const id = typeof idOrCtrl === "object" ? idOrCtrl.id : idOrCtrl;
      const idx = this.controls.findIndex((c) => c.id === id);
      if (idx !== -1) {
        const [ctrl] = this.controls.splice(idx, 1);
        if (ctrl.el && ctrl.el.parentNode) ctrl.el.parentNode.removeChild(ctrl.el);
        this.syncModalState();
        this.wakeUp();
      }
    }

    clear() {
      while (this.controls.length) {
        const c = this.controls.pop();
        if (c.el && c.el.parentNode) c.el.parentNode.removeChild(c.el);
      }
      this.syncModalState();
      this.wakeUp();
    }

    getById(id) {
      return this.controls.find((c) => c.id === id) || null;
    }

    measureAndSyncToolbarPill(ctrl, immediate = false) {
      if (ctrl.type !== "toolbar" || !ctrl.toolbarItemEls || !ctrl.toolbarItemEls.length) return;
      let totalNeededW = 20;
      for (let i = 0; i < ctrl.toolbarItemEls.length; i++) {
        totalNeededW += ctrl.toolbarItemEls[i].offsetWidth + 4;
      }
      if (ctrl.targetWidth < totalNeededW) {
        ctrl.targetWidth = totalNeededW;
        if (immediate) ctrl.width = totalNeededW;
      }

      const activeItem = ctrl.toolbarItemEls[ctrl.selectedIndex] || ctrl.toolbarItemEls[0];
      if (!activeItem) return;

      const itemCenterLocalX = activeItem.offsetLeft + activeItem.offsetWidth * 0.5 - ctrl.width * 0.5;
      const itemHalfW = Math.max(28, activeItem.offsetWidth * 0.5 + 2);
      const itemHalfH = Math.max(15, activeItem.offsetHeight * 0.5 + 1);

      ctrl.targetPillOffsetX = itemCenterLocalX;
      ctrl.targetPillHalfW = itemHalfW;
      ctrl.targetPillHalfH = itemHalfH;

      if (immediate) {
        ctrl.pillOffsetX = itemCenterLocalX;
        ctrl.pillHalfW = itemHalfW;
        ctrl.pillHalfH = itemHalfH;
      }
    }

    setupControlInnerDOM(ctrl) {
      const el = ctrl.el;
      el.innerHTML = "";

      if (ctrl.type === "button" || ctrl.type === "text") {
        const span = document.createElement("span");
        span.className = "lg-btn-label";
        span.textContent = ctrl.label;
        el.appendChild(span);
        ctrl.labelSpan = span;
      } else if (ctrl.type === "toolbar") {
        ctrl.pillOffsetX = 0;
        ctrl.targetPillOffsetX = 0;
        ctrl.vPillOffsetX = 0;
        ctrl.pillHalfW = 42;
        ctrl.targetPillHalfW = 42;
        ctrl.vPillHalfW = 0;
        ctrl.pillHalfH = 18;
        ctrl.targetPillHalfH = 18;
        ctrl.pillScale = 1.0;
        ctrl.targetPillScale = 1.0;
        ctrl.vPillScale = 0;
        ctrl.pillGlow = 0;
        ctrl.targetPillGlow = 0;
        ctrl.vPillGlow = 0;
        ctrl.toolbarItemEls = [];

        ctrl.options.forEach((txt, idx) => {
          const item = document.createElement("div");
          item.className = "lg-toolbar-item" + (idx === ctrl.selectedIndex ? " active" : "");
          item.textContent = txt;
          el.appendChild(item);
          ctrl.toolbarItemEls.push(item);
        });
      } else if (ctrl.type === "dropdown") {
        const span = document.createElement("span");
        span.className = "lg-dropdown-text";
        span.textContent = ctrl.label;
        const arrow = document.createElement("span");
        arrow.className = "lg-dropdown-arrow";
        arrow.textContent = "▾";
        el.append(span, arrow);
        ctrl.labelSpan = span;
      } else if (ctrl.type === "slider") {
        ctrl.glassActive = 0;
        const trackBg = document.createElement("div");
        trackBg.className = "lg-slider-track-bg";
        const trackFill = document.createElement("div");
        trackFill.className = "lg-slider-track-fill";
        trackBg.appendChild(trackFill);

        const thumbDom = document.createElement("div");
        thumbDom.className = "lg-slider-thumb-dom";
        el.append(trackBg, thumbDom);

        ctrl.trackBgEl = trackBg;
        ctrl.trackFillEl = trackFill;
        ctrl.thumbDomEl = thumbDom;
        ctrl.thumbScale = 1;
        ctrl.targetThumbScale = 1;
        ctrl.vThumbScale = 0;
        ctrl.thumbGlassMix = 0;
        ctrl.targetThumbGlassMix = 0;
        ctrl.vThumbGlassMix = 0;
      } else if (ctrl.type === "switch") {
        ctrl.glassActive = 0;
        const trackDom = document.createElement("div");
        trackDom.className = "lg-switch-track-dom";
        const thumbDom = document.createElement("div");
        thumbDom.className = "lg-switch-thumb-dom";
        el.append(trackDom, thumbDom);

        ctrl.switchTrackEl = trackDom;
        ctrl.thumbDomEl = thumbDom;
        ctrl.thumbX = ctrl.checked ? 1 : 0;
        ctrl.targetThumbX = ctrl.checked ? 1 : 0;
        ctrl.vThumbX = 0;
        ctrl.thumbScale = 1;
        ctrl.targetThumbScale = 1;
        ctrl.vThumbScale = 0;
        ctrl.thumbGlassMix = 0;
        ctrl.targetThumbGlassMix = 0;
        ctrl.vThumbGlassMix = 0;
      } else if (ctrl.type === "editor") {
        const header = document.createElement("div");
        header.className = "lg-editor-header";
        header.innerHTML = `<span>✦ ${ctrl.description || "Liquid Glass Editor"}</span><span>UTF-8</span>`;
        const ta = document.createElement("textarea");
        ta.className = "lg-editor-content";
        ta.value = ctrl.label;
        ta.addEventListener("input", () => {
          ctrl.label = ta.value;
          ctrl.targetGlow = 1.25;
          ctrl.vScale += 0.55;
          ctrl.emit("input", ta.value);
          this.wakeUp();
          clearTimeout(ctrl._editorGlowTimer);
          ctrl._editorGlowTimer = setTimeout(() => {
            if (!ctrl.isPressed) {
              ctrl.targetGlow = 0;
              this.wakeUp();
            }
          }, 260);
        });
        el.append(header, ta);
        ctrl.textareaEl = ta;
      } else if (ctrl.type === "menu") {
        const mask = document.createElement("div");
        mask.className = "lg-menu-highlight-mask";
        el.appendChild(mask);
        ctrl.menuMaskEl = mask;
        ctrl.maskX = 12;
        ctrl.maskY = 10;
        ctrl.maskW = 120;
        ctrl.maskH = 32;
        ctrl.targetMaskX = 12;
        ctrl.targetMaskY = 10;
        ctrl.targetMaskW = 120;
        ctrl.targetMaskH = 32;
        ctrl.vMaskX = 0;
        ctrl.vMaskY = 0;
        ctrl.vMaskW = 0;
        ctrl.vMaskH = 0;
        ctrl.maskOpacity = 0;
        ctrl.targetMaskOpacity = 0;
        ctrl.vMaskOpacity = 0;

        ctrl.menuItemEls = [];
        ctrl.options.forEach((optText, idx) => {
          const item = document.createElement("div");
          item.className = "lg-menu-item";
          const check = document.createElement("span");
          check.className = "lg-menu-check";
          check.textContent = idx === ctrl.selectedIndex ? "✓" : "";
          const textNode = document.createElement("span");
          textNode.textContent = optText;
          item.append(check, textNode);
          el.appendChild(item);
          ctrl.menuItemEls.push({ item, check, textNode, idx });
        });
      } else if (ctrl.type === "div") {
        el.innerHTML = `
          <div class="lg-div-title">${ctrl.label}</div>
          <div class="lg-div-body">${ctrl.description || ""}</div>
        `;
      } else if (ctrl.type === "alert") {
        el.innerHTML = `
          <div class="lg-alert-badge">System Optical Alert</div>
          <div class="lg-alert-title">${ctrl.label}</div>
          <div class="lg-alert-desc">${ctrl.description || ""}</div>
          <div class="lg-alert-actions">
            <div class="lg-alert-btn" data-act="cancel">收拢通知</div>
            <div class="lg-alert-btn primary" data-act="confirm">确认</div>
          </div>
        `;
        el.querySelectorAll(".lg-alert-btn").forEach((btn) => {
          btn.addEventListener("pointerdown", (ev) => {
            ev.stopPropagation();
            if (!ctrl.interactive) return;
            const act = btn.dataset.act;
            ctrl.emit(act);
            if (ctrl.morphTargetId) {
              this.triggerContextualMorph(ctrl, this.getById(ctrl.morphTargetId));
            } else {
              ctrl.hide();
            }
          });
        });
      }
    }

    bindControlEvents(ctrl) {
      const el = ctrl.el;

      const updateToolbarSelectionFromPointer = (clientX) => {
        if (ctrl.type !== "toolbar" || !ctrl.toolbarItemEls) return;
        let bestIdx = ctrl.selectedIndex;
        let minDist = Infinity;
        for (let i = 0; i < ctrl.toolbarItemEls.length; i++) {
          const r = ctrl.toolbarItemEls[i].getBoundingClientRect();
          const d = Math.abs(clientX - (r.left + r.width * 0.5));
          if (d < minDist) {
            minDist = d;
            bestIdx = i;
          }
        }
        if (bestIdx !== ctrl.selectedIndex) {
          ctrl.selectedIndex = bestIdx;
          ctrl.toolbarItemEls.forEach((item, idx) => item.classList.toggle("active", idx === bestIdx));
          this.measureAndSyncToolbarPill(ctrl, false);
          ctrl.vPillScale += 1.65;
          ctrl.emit("change", bestIdx, ctrl.options[bestIdx]);
        }
      };

      const cacheMenuMetrics = () => {
        if (ctrl.type !== "menu") return;
        const rect = el.getBoundingClientRect();
        const invScale = 1 / Math.max(0.2, ctrl.scale);
        ctrl._cachedMenuItems = ctrl.menuItemEls.map((entry) => {
          const r = entry.item.getBoundingClientRect();
          return {
            idx: entry.idx,
            cy: r.top + r.height * 0.5,
            localX: (r.left - rect.left) * invScale,
            localY: (r.top - rect.top) * invScale,
            localW: Math.max(64, r.width * invScale + 6),
            localH: r.height * invScale
          };
        });
      };

      const updateMenuMaskFromPointer = (clientY) => {
        if (ctrl.type !== "menu" || !ctrl._cachedMenuItems) return;
        let best = null;
        let minDist = Infinity;
        for (let i = 0; i < ctrl._cachedMenuItems.length; i++) {
          const item = ctrl._cachedMenuItems[i];
          const d = Math.abs(clientY - item.cy);
          if (d < minDist) {
            minDist = d;
            best = item;
          }
        }
        if (best) {
          ctrl.hoveredMenuIdx = best.idx;
          ctrl.targetMaskX = best.localX;
          ctrl.targetMaskY = best.localY;
          ctrl.targetMaskW = best.localW;
          ctrl.targetMaskH = best.localH;
          ctrl.targetMaskOpacity = 1;
        }
      };

      el.addEventListener("pointerdown", (e) => {
        if (!ctrl.interactive || ctrl.hiddenByMorph) return;
        ctrl.zOrder = this.zOrderCounter++;
        this.wakeUp();

        ctrl.isPressed = true;
        ctrl.pointerStartX = e.clientX;
        ctrl.pointerStartY = e.clientY;
        ctrl.originX = ctrl.targetX;
        ctrl.originY = ctrl.targetY;
        ctrl.touchOffsetX = e.clientX - ctrl.x;
        ctrl.touchOffsetY = e.clientY - ctrl.y;
        ctrl.lastPointerX = e.clientX;
        ctrl.lastPointerTime = performance.now();
        ctrl._cachedRect = el.getBoundingClientRect();

        ctrl.targetScale = 1.07;
        ctrl.targetGlow = 1.22;

        if (ctrl.type === "editor" && e.target === ctrl.textareaEl && !e.altKey) {
          ctrl.editingTextMode = true;
          ctrl.targetScale = 1.02;
          return;
        }
        ctrl.editingTextMode = false;
        el.setPointerCapture(e.pointerId);

        if (ctrl.type === "toolbar") {
          ctrl.targetScale = 1.02;
          ctrl.targetPillScale = 1.16;
          ctrl.targetPillGlow = 1.15;
          updateToolbarSelectionFromPointer(e.clientX);
        } else if (ctrl.type === "slider") {
          ctrl.targetScale = 1.0;
          ctrl.targetThumbScale = 1.38;
          ctrl.targetThumbGlassMix = 1.0;
          const rect = ctrl._cachedRect;
          const pad = 20;
          const raw = (e.clientX - (rect.left + pad)) / Math.max(1, rect.width - pad * 2);
          ctrl.value = Math.max(-0.08, Math.min(1.08, raw));
          ctrl.emit("change", Math.max(0, Math.min(1, ctrl.value)));
        } else if (ctrl.type === "switch") {
          ctrl.targetScale = 1.03;
          ctrl.targetThumbScale = 1.36;
          ctrl.targetThumbGlassMix = 1.0;
        } else if (ctrl.type === "menu") {
          cacheMenuMetrics();
          updateMenuMaskFromPointer(e.clientY);
          ctrl.maskX = ctrl.targetMaskX;
          ctrl.maskY = ctrl.targetMaskY;
          ctrl.maskW = ctrl.targetMaskW;
          ctrl.maskH = ctrl.targetMaskH;
        }
      });

      el.addEventListener("pointermove", (e) => {
        if (!ctrl.interactive || !ctrl.isPressed) return;
        this.wakeUp();
        ctrl.touchOffsetX = e.clientX - ctrl.x;
        ctrl.touchOffsetY = e.clientY - ctrl.y;

        if (ctrl.editingTextMode) return;

        const dx = e.clientX - ctrl.pointerStartX;
        const dy = e.clientY - ctrl.pointerStartY;

        if (ctrl.type === "toolbar") {
          updateToolbarSelectionFromPointer(e.clientX);
          if (ctrl.draggableDeform && Math.hypot(dx, dy) > 18) {
            ctrl.targetX = ctrl.originX + dx;
            ctrl.targetY = ctrl.originY + dy;
          }
          return;
        }

        if (ctrl.type === "slider") {
          const rect = ctrl._cachedRect;
          const now = performance.now();
          const dt = Math.max(1, now - ctrl.lastPointerTime);
          const vxPx = (e.clientX - ctrl.lastPointerX) / dt;
          ctrl.lastPointerX = e.clientX;
          ctrl.lastPointerTime = now;

          const pad = 20;
          const raw = (e.clientX - (rect.left + pad)) / Math.max(1, rect.width - pad * 2);
          ctrl.value = raw < 0 ? raw * 0.22 : (raw > 1 ? 1 + (raw - 1) * 0.22 : raw);
          ctrl.vValue = vxPx * 0.85;
          ctrl.targetSkewX = Math.max(-14, Math.min(14, vxPx * 7));
          ctrl.emit("change", Math.max(0, Math.min(1, ctrl.value)));
          return;
        }

        if (ctrl.type === "switch") {
          const rect = ctrl._cachedRect;
          const rel = (e.clientX - rect.left) / Math.max(1, rect.width);
          ctrl.targetThumbX = Math.max(0, Math.min(1, rel));
          ctrl.targetColorMix = ctrl.targetThumbX;
          return;
        }

        if (ctrl.type === "menu") {
          updateMenuMaskFromPointer(e.clientY);
        }

        if (ctrl.draggableDeform) {
          ctrl.targetX = ctrl.originX + dx;
          ctrl.targetY = ctrl.originY + dy;
          ctrl.targetSkewX = Math.max(-18, Math.min(18, dx * 0.1));
          ctrl.targetSkewY = Math.max(-18, Math.min(18, dy * 0.1));
          ctrl.emit("drag", { x: ctrl.targetX, y: ctrl.targetY });
        }
      });

      const finishPointer = (e) => {
        if (!ctrl.isPressed) return;
        this.wakeUp();
        ctrl.isPressed = false;
        ctrl.editingTextMode = false;
        ctrl.targetScale = 1.0;
        ctrl.targetGlow = 0.0;
        ctrl.targetSkewX = 0;
        ctrl.targetSkewY = 0;

        const moveDist = Math.hypot(e.clientX - ctrl.pointerStartX, e.clientY - ctrl.pointerStartY);
        if (moveDist < 8) ctrl.emit("click", e);

        if (ctrl.type === "toolbar") {
          ctrl.targetPillScale = 1.0;
          ctrl.targetPillGlow = 0.0;
        } else if (ctrl.type === "slider") {
          ctrl.targetThumbScale = 1.0;
          ctrl.targetThumbGlassMix = 0.0;
        } else if (ctrl.type === "switch") {
          ctrl.targetThumbScale = 1.0;
          ctrl.targetThumbGlassMix = 0.0;
          ctrl.checked = moveDist < 8 ? !ctrl.checked : ctrl.targetThumbX >= 0.5;
          ctrl.targetThumbX = ctrl.checked ? 1 : 0;
          ctrl.targetColorMix = ctrl.checked ? 1 : 0;
          ctrl.emit("change", ctrl.checked);
        } else if (ctrl.type === "menu") {
          if (ctrl.hoveredMenuIdx !== undefined) {
            ctrl.selectedIndex = ctrl.hoveredMenuIdx;
            ctrl.menuItemEls.forEach((entry) => {
              entry.check.textContent = entry.idx === ctrl.selectedIndex ? "✓" : "";
            });
            ctrl.emit("change", ctrl.selectedIndex, ctrl.options[ctrl.selectedIndex]);
            if (ctrl.morphTargetId && moveDist < 24) {
              const targetCtrl = this.getById(ctrl.morphTargetId);
              if (targetCtrl && targetCtrl.labelSpan) {
                targetCtrl.labelSpan.textContent = "预设: " + ctrl.options[ctrl.selectedIndex];
              }
              setTimeout(() => this.triggerContextualMorph(ctrl, targetCtrl), 140);
            }
          }
          ctrl.targetMaskOpacity = 0;
        } else if (ctrl.morphTargetId && moveDist < 10 && ctrl.type !== "alert") {
          const target = this.getById(ctrl.morphTargetId);
          if (target) this.triggerContextualMorph(ctrl, target);
        }
      };

      el.addEventListener("pointerup", finishPointer);
      el.addEventListener("pointercancel", finishPointer);
    }

    triggerContextualMorph(fromCtrl, toCtrl) {
      if (!fromCtrl || !toCtrl) return;

      if (toCtrl.type === "alert") {
        toCtrl.targetX = (this.stageWidth || window.innerWidth) * 0.5;
        toCtrl.targetY = (this.stageHeight || window.innerHeight) * 0.5;
      }

      const destX = toCtrl.targetX;
      const destY = toCtrl.targetY;
      const destW = toCtrl.targetWidth;
      const destH = toCtrl.targetHeight;
      const destR = toCtrl.targetRadius;

      toCtrl.x = fromCtrl.x;
      toCtrl.y = fromCtrl.y;
      toCtrl.width = fromCtrl.width;
      toCtrl.height = fromCtrl.height;
      toCtrl.borderRadius = fromCtrl.borderRadius;
      toCtrl.scale = 0.92;
      toCtrl.vScale = 2.8;
      toCtrl.skewX = Math.max(-20, Math.min(20, (destX - fromCtrl.x) * 0.08));
      toCtrl.skewY = Math.max(-20, Math.min(20, (destY - fromCtrl.y) * 0.08));
      toCtrl.glowIntensity = 1.35;
      toCtrl.targetGlow = 0.0;

      toCtrl.targetX = destX;
      toCtrl.targetY = destY;
      toCtrl.targetWidth = destW;
      toCtrl.targetHeight = destH;
      toCtrl.targetRadius = destR;
      toCtrl.targetScale = 1.0;
      toCtrl.hiddenByMorph = false;
      toCtrl.glassActive = 1;
      toCtrl.zOrder = this.zOrderCounter++;
      toCtrl.el.style.opacity = "1";
      toCtrl.el.style.pointerEvents = toCtrl.interactive ? "auto" : "none";

      fromCtrl.hiddenByMorph = true;
      fromCtrl.targetScale = 0.35;
      fromCtrl.glassActive = 0;
      fromCtrl.el.style.opacity = "0";
      fromCtrl.el.style.pointerEvents = "none";

      fromCtrl.emit("morph", toCtrl);
      this.syncModalState();
      this.wakeUp();
    }

    syncModalState() {
      const isAlertOpen = this.controls.some((c) => c.type === "alert" && !c.hiddenByMorph);
      this.targetModalDim = isAlertOpen ? 1.0 : 0.0;
      this.stage.classList.toggle("lg-modal-open", isAlertOpen);
      if (this.modalBackdrop) {
        this.modalBackdrop.classList.toggle("active", isAlertOpen);
      }
    }

    /* ==========================================================================
     * 节流脏检查版 DOM 遮罩系统 (仅在控件位移超过 0.5px 时重算 SVG Mask)
     * ========================================================================== */
    updateDOMOcclusionMasks() {
      let moved = this.occlusionDirty;
      const len = this.controls.length;
      for (let i = 0; i < len; i++) {
        const c = this.controls[i];
        if (Math.abs(c.x - c._lastMaskX) > 0.6 || Math.abs(c.y - c._lastMaskY) > 0.6) {
          c._lastMaskX = c.x;
          c._lastMaskY = c.y;
          moved = true;
        }
      }
      if (!moved) return;
      this.occlusionDirty = false;

      const visible = [];
      for (let i = 0; i < len; i++) {
        const c = this.controls[i];
        if (!c.hiddenByMorph && c.el.style.opacity !== "0") visible.push(c);
      }
      visible.sort((a, b) => (a.layerId - b.layerId) || (a.zOrder - b.zOrder));

      for (let i = 0; i < visible.length; i++) {
        const low = visible[i];
        low.el.style.zIndex = String(Math.round(low.layerId * 100) + i);

        const lW = low.width * low.scale;
        const lH = low.height * low.scale;
        const lLeft = low.x - lW * 0.5;
        const lTop = low.y - lH * 0.5;
        const lRight = lLeft + lW;
        const lBottom = lTop + lH;

        let key = "";
        const rects = [];
        for (let j = i + 1; j < visible.length; j++) {
          const high = visible[j];
          if (high.type === "slider" || high.type === "switch") continue;

          const hW = high.width * high.scale;
          const hH = high.height * high.scale;
          const hLeft = high.x - hW * 0.5;
          const hTop = high.y - hH * 0.5;

          if (hLeft < lRight - 2 && (hLeft + hW) > lLeft + 2 && hTop < lBottom - 2 && (hTop + hH) > lTop + 2) {
            const invScale = 1 / Math.max(0.2, low.scale);
            const rx = Math.round((hLeft - lLeft) * invScale);
            const ry = Math.round((hTop - lTop) * invScale);
            const rw = Math.round(hW * invScale);
            const rh = Math.round(hH * invScale);
            const rr = Math.round(high.borderRadius * invScale);
            key += `${rx},${ry},${rw},${rh},${rr}|`;
            rects.push(`<rect x="${rx}" y="${ry}" width="${rw}" height="${rh}" rx="${rr}" fill="black" fill-opacity="0.92"/>`);
          }
        }

        if (!key) {
          if (low._lastMaskKey !== "none") {
            low.el.style.webkitMaskImage = "none";
            low.el.style.maskImage = "none";
            low._lastMaskKey = "none";
          }
        } else if (key !== low._lastMaskKey) {
          const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="${Math.round(low.width)}" height="${Math.round(low.height)}"><rect width="100%" height="100%" fill="white"/>${rects.join("")}</svg>`;
          const url = `url("data:image/svg+xml;utf8,${encodeURIComponent(svg)}")`;
          low.el.style.webkitMaskImage = url;
          low.el.style.maskImage = url;
          low._lastMaskKey = key;
        }
      }
    }

    /* ==========================================================================
     * 零 GC 主循环 + 动能收敛自动休眠
     * ========================================================================== */
    frameLoop(now) {
      const rawDt = Math.min(0.032, (now - (this.lastFrameTime || now)) / 1000 || 0.016);
      this.lastFrameTime = now;
      const elapsed = (now - this.startTime) / 1000;
      this.frameCount++;

      this.updateBackgroundTexture(elapsed, false);

      // 每 10 帧使用预分配 Float32Array 向 Worker 投递粘连检测
      if (this.worker && this.frameCount % 10 === 0) {
        let wCount = 0;
        for (let i = 0; i < this.controls.length; i++) {
          const c = this.controls[i];
          if (c.adhesion && c.glassActive > 0.1 && wCount < this.maxControls) {
            const o = wCount * 6;
            this.workerTransferBuf[o] = c.id;
            this.workerTransferBuf[o + 1] = c.x;
            this.workerTransferBuf[o + 2] = c.y;
            this.workerTransferBuf[o + 3] = c.width * c.scale;
            this.workerTransferBuf[o + 4] = c.height * c.scale;
            this.workerTransferBuf[o + 5] = c.layerId;
            wCount++;
          }
        }
        this.worker.postMessage({ type: "step", buffer: this.workerTransferBuf, count: wCount });
      }

      const stepPos = this.wasm.stepPos;
      const stepVel = this.wasm.stepVel;
      let totalEnergy = 0;

      // 陀螺仪弹簧平滑
      const gx = stepPos(this.gyroX, this.vGyroX, this.targetGyroX, 140.0, 14.0, rawDt);
      this.vGyroX = stepVel(this.gyroX, this.vGyroX, this.targetGyroX, 140.0, 14.0, rawDt);
      this.gyroX = gx;

      const gy = stepPos(this.gyroY, this.vGyroY, this.targetGyroY, 140.0, 14.0, rawDt);
      this.vGyroY = stepVel(this.gyroY, this.vGyroY, this.targetGyroY, 140.0, 14.0, rawDt);
      this.gyroY = gy;

      this.targetGyroBoost *= Math.pow(0.05, rawDt);
      const gb = stepPos(this.gyroBoost, this.vGyroBoost, this.targetGyroBoost, 180.0, 16.0, rawDt);
      this.vGyroBoost = stepVel(this.gyroBoost, this.vGyroBoost, this.targetGyroBoost, 180.0, 16.0, rawDt);
      this.gyroBoost = Math.max(0, gb);

      const md = stepPos(this.modalDim, this.vModalDim, this.targetModalDim, 160.0, 18.0, rawDt);
      this.vModalDim = stepVel(this.modalDim, this.vModalDim, this.targetModalDim, 160.0, 18.0, rawDt);
      this.modalDim = Math.max(0, Math.min(1, md));

      totalEnergy += Math.abs(this.vGyroX) + Math.abs(this.vGyroY) + Math.abs(this.vModalDim) + this.gyroBoost;

      const subSteps = 2;
      const dt = rawDt / subSteps;
      const kSpring = 210.0;
      const dSpring = 16.5;

      const trackList = this._trackPrimitives;
      const glassList = this._glassPrimitives;
      trackList.length = 0;
      glassList.length = 0;

      const hasWebGPU = Boolean(this.device && this.pipeline);

      for (let idx = 0; idx < this.controls.length; idx++) {
        const c = this.controls[idx];
        for (let s = 0; s < subSteps; s++) {
          const nx = stepPos(c.x, c.vx, c.targetX, kSpring, dSpring, dt);
          c.vx = stepVel(c.x, c.vx, c.targetX, kSpring, dSpring, dt);
          c.x = nx;

          const ny = stepPos(c.y, c.vy, c.targetY, kSpring, dSpring, dt);
          c.vy = stepVel(c.y, c.vy, c.targetY, kSpring, dSpring, dt);
          c.y = ny;

          const nw = stepPos(c.width, c.vw, c.targetWidth, kSpring, dSpring, dt);
          c.vw = stepVel(c.width, c.vw, c.targetWidth, kSpring, dSpring, dt);
          c.width = nw;

          const nh = stepPos(c.height, c.vh, c.targetHeight, kSpring, dSpring, dt);
          c.vh = stepVel(c.height, c.vh, c.targetHeight, kSpring, dSpring, dt);
          c.height = nh;

          const nr = stepPos(c.borderRadius, c.vRadius, c.targetRadius, kSpring, dSpring, dt);
          c.vRadius = stepVel(c.borderRadius, c.vRadius, c.targetRadius, kSpring, dSpring, dt);
          c.borderRadius = nr;

          const ns = stepPos(c.scale, c.vScale, c.targetScale, 240.0, 14.5, dt);
          c.vScale = stepVel(c.scale, c.vScale, c.targetScale, 240.0, 14.5, dt);
          c.scale = ns;

          const ng = stepPos(c.glowIntensity, c.vGlow, c.targetGlow, 180.0, 18.0, dt);
          c.vGlow = stepVel(c.glowIntensity, c.vGlow, c.targetGlow, 180.0, 18.0, dt);
          c.glowIntensity = Math.max(0, ng);

          const nsx = stepPos(c.skewX, c.vSkewX, c.targetSkewX, 190.0, 15.0, dt);
          c.vSkewX = stepVel(c.skewX, c.vSkewX, c.targetSkewX, 190.0, 15.0, dt);
          c.skewX = nsx;

          const nsy = stepPos(c.skewY, c.vSkewY, c.targetSkewY, 190.0, 15.0, dt);
          c.vSkewY = stepVel(c.skewY, c.vSkewY, c.targetSkewY, 190.0, 15.0, dt);
          c.skewY = nsy;
        }

        totalEnergy += Math.abs(c.vx) + Math.abs(c.vy) + Math.abs(c.vw) + Math.abs(c.vh) + Math.abs(c.vScale) + Math.abs(c.vGlow) + (c.isPressed ? 1 : 0);

        const left = c.x - c.width * 0.5;
        const top = c.y - c.height * 0.5;
        if (Math.abs(c.width - c._lastCssW) > 0.05) {
          c.el.style.width = `${c.width.toFixed(1)}px`;
          c._lastCssW = c.width;
        }
        if (Math.abs(c.height - c._lastCssH) > 0.05) {
          c.el.style.height = `${c.height.toFixed(1)}px`;
          c._lastCssH = c.height;
        }
        if (Math.abs(c.borderRadius - c._lastCssR) > 0.05) {
          c.el.style.borderRadius = `${c.borderRadius.toFixed(1)}px`;
          c._lastCssR = c.borderRadius;
        }
        const tStr = `translate3d(${left.toFixed(2)}px,${top.toFixed(2)}px,0) scale(${c.scale.toFixed(4)}) skew(${(c.skewX * 0.25).toFixed(2)}deg,${(c.skewY * 0.25).toFixed(2)}deg)`;
        if (tStr !== c._lastCssT) {
          c.el.style.transform = tStr;
          c._lastCssT = tStr;
        }

        const worldTouchX = c.x + c.touchOffsetX;
        const worldTouchY = c.y + c.touchOffsetY;

        if (c.type === "menu" && c.menuMaskEl) {
          c.maskX = stepPos(c.maskX, c.vMaskX, c.targetMaskX, 260, 18, rawDt);
          c.vMaskX = stepVel(c.maskX, c.vMaskX, c.targetMaskX, 260, 18, rawDt);
          c.maskY = stepPos(c.maskY, c.vMaskY, c.targetMaskY, 260, 18, rawDt);
          c.vMaskY = stepVel(c.maskY, c.vMaskY, c.targetMaskY, 260, 18, rawDt);
          c.maskW = stepPos(c.maskW, c.vMaskW, c.targetMaskW, 250, 17, rawDt);
          c.vMaskW = stepVel(c.maskW, c.vMaskW, c.targetMaskW, 250, 17, rawDt);
          c.maskH = stepPos(c.maskH, c.vMaskH, c.targetMaskH, 250, 17, rawDt);
          c.vMaskH = stepVel(c.maskH, c.vMaskH, c.targetMaskH, 250, 17, rawDt);
          c.maskOpacity = stepPos(c.maskOpacity, c.vMaskOpacity, c.targetMaskOpacity, 200, 20, rawDt);
          c.vMaskOpacity = stepVel(c.maskOpacity, c.vMaskOpacity, c.targetMaskOpacity, 200, 20, rawDt);

          totalEnergy += Math.abs(c.vMaskX) + Math.abs(c.vMaskY) + Math.abs(c.vMaskW) + Math.abs(c.vMaskOpacity);

          if (c.maskOpacity > 0.005) {
            c.menuMaskEl.style.width = `${Math.max(10, c.maskW).toFixed(1)}px`;
            c.menuMaskEl.style.height = `${Math.max(10, c.maskH).toFixed(1)}px`;
            c.menuMaskEl.style.transform = `translate3d(${c.maskX.toFixed(1)}px, ${c.maskY.toFixed(1)}px, 0)`;
          }
          c.menuMaskEl.style.opacity = String(Math.max(0, Math.min(1, c.maskOpacity)));
        }

        if (c.type === "slider") {
          if (!c.isPressed) {
            c.value += c.vValue * rawDt;
            c.vValue *= Math.pow(0.04, rawDt);
            if (c.value < 0) {
              const nv = stepPos(c.value, c.vValue, 0, 220, 16, rawDt);
              c.vValue = stepVel(c.value, c.vValue, 0, 220, 16, rawDt);
              c.value = nv;
            } else if (c.value > 1) {
              const nv = stepPos(c.value, c.vValue, 1, 220, 16, rawDt);
              c.vValue = stepVel(c.value, c.vValue, 1, 220, 16, rawDt);
              c.value = nv;
            }
          }

          const ts = stepPos(c.thumbScale, c.vThumbScale, c.targetThumbScale, 240, 15, rawDt);
          c.vThumbScale = stepVel(c.thumbScale, c.vThumbScale, c.targetThumbScale, 240, 15, rawDt);
          c.thumbScale = ts;

          const gm = stepPos(c.thumbGlassMix, c.vThumbGlassMix, c.targetThumbGlassMix, 220, 18, rawDt);
          c.vThumbGlassMix = stepVel(c.thumbGlassMix, c.vThumbGlassMix, c.targetThumbGlassMix, 220, 18, rawDt);
          c.thumbGlassMix = Math.max(0, Math.min(1, gm));

          totalEnergy += Math.abs(c.vValue) + Math.abs(c.vThumbScale) + Math.abs(c.vThumbGlassMix);

          const clampedFill = Math.max(0, Math.min(1, c.value));
          const pad = 20;
          const usableW = c.width - pad * 2;
          const thumbLocalX = pad + c.value * usableW;
          c.thumbDomEl.style.left = `${thumbLocalX.toFixed(2)}px`;
          c.thumbDomEl.style.transform = `translate(-50%, -50%) scale(${c.thumbScale.toFixed(3)})`;
          c.thumbDomEl.style.opacity = String(hasWebGPU ? Math.max(0, 1 - c.thumbGlassMix) : 1);

          const trackRgb = hexToRgb01(c.tint ? c.tintColor : "#8e8e93");
          trackList.push({
            cx: c.x * this.dpr,
            cy: c.y * this.dpr,
            hx: (c.width * 0.5 - pad) * this.dpr,
            hy: 4.5 * this.dpr,
            radius: 4.5 * this.dpr,
            translucency: 0,
            tintEnabled: 1,
            tintOpacity: 1,
            tintColor: trackRgb,
            adhesion: 0,
            tx: worldTouchX * this.dpr,
            ty: worldTouchY * this.dpr,
            glow: 0,
            layer: c.layerId,
            skewX: 0,
            skewY: 0,
            glassActive: 0,
            solidTrack: 1.0,
            trackFillRatio: clampedFill,
            trackColor: trackRgb
          });

          if (c.thumbGlassMix > 0.01) {
            const worldThumbX = c.x - c.width * 0.5 + thumbLocalX;
            const velocityStretch = Math.min(5.5, Math.abs(c.vValue) * 3.5);
            const capHalfW = (20.0 * c.thumbScale + velocityStretch) * this.dpr;
            const capHalfH = (12.0 * c.thumbScale) * this.dpr;
            glassList.push({
              cx: worldThumbX * this.dpr,
              cy: c.y * this.dpr,
              hx: capHalfW,
              hy: capHalfH,
              radius: capHalfH,
              translucency: c.translucency,
              tintEnabled: 0.0,
              tintOpacity: 0.0,
              tintColor: [1.0, 1.0, 1.0],
              adhesion: 0,
              tx: worldThumbX * this.dpr,
              ty: c.y * this.dpr,
              glow: 0.55 * c.thumbGlassMix,
              layer: c.layerId,
              skewX: c.skewX * 0.6 * this.dpr,
              skewY: 0,
              glassActive: c.thumbGlassMix,
              solidTrack: 0,
              trackFillRatio: 0,
              trackColor: trackRgb
            });
          }
          continue;
        }

        if (c.type === "switch") {
          const ntx = stepPos(c.thumbX, c.vThumbX, c.targetThumbX, 230, 17, rawDt);
          c.vThumbX = stepVel(c.thumbX, c.vThumbX, c.targetThumbX, 230, 17, rawDt);
          c.thumbX = ntx;

          const ncm = stepPos(c.colorMix, c.vColorMix, c.targetColorMix, 180, 18, rawDt);
          c.vColorMix = stepVel(c.colorMix, c.vColorMix, c.targetColorMix, 180, 18, rawDt);
          c.colorMix = Math.max(0, Math.min(1, ncm));

          const ts = stepPos(c.thumbScale, c.vThumbScale, c.targetThumbScale, 240, 15, rawDt);
          c.vThumbScale = stepVel(c.thumbScale, c.vThumbScale, c.targetThumbScale, 240, 15, rawDt);
          c.thumbScale = ts;

          const gm = stepPos(c.thumbGlassMix, c.vThumbGlassMix, c.targetThumbGlassMix, 220, 18, rawDt);
          c.vThumbGlassMix = stepVel(c.thumbGlassMix, c.vThumbGlassMix, c.targetThumbGlassMix, 220, 18, rawDt);
          c.thumbGlassMix = Math.max(0, Math.min(1, gm));

          totalEnergy += Math.abs(c.vThumbX) + Math.abs(c.vColorMix) + Math.abs(c.vThumbScale) + Math.abs(c.vThumbGlassMix);

          const offRgb = [0.30, 0.32, 0.36];
          const onRgb = hexToRgb01(c.tint ? c.tintColor : "#34c759");
          const curRgb = [
            offRgb[0] + (onRgb[0] - offRgb[0]) * c.colorMix,
            offRgb[1] + (onRgb[1] - offRgb[1]) * c.colorMix,
            offRgb[2] + (onRgb[2] - offRgb[2]) * c.colorMix
          ];

          const pad = 23;
          const thumbLocalX = pad + c.thumbX * (c.width - pad * 2);
          c.thumbDomEl.style.left = `${thumbLocalX.toFixed(2)}px`;
          c.thumbDomEl.style.transform = `translate(-50%, -50%) scale(${c.thumbScale.toFixed(3)})`;
          c.thumbDomEl.style.opacity = String(hasWebGPU ? Math.max(0, 1 - c.thumbGlassMix) : 1);

          trackList.push({
            cx: c.x * this.dpr,
            cy: c.y * this.dpr,
            hx: c.width * 0.5 * this.dpr,
            hy: c.height * 0.5 * this.dpr,
            radius: c.borderRadius * this.dpr,
            translucency: 0,
            tintEnabled: 1,
            tintOpacity: 1,
            tintColor: curRgb,
            adhesion: 0,
            tx: worldTouchX * this.dpr,
            ty: worldTouchY * this.dpr,
            glow: 0,
            layer: c.layerId,
            skewX: 0,
            skewY: 0,
            glassActive: 0,
            solidTrack: 2.0,
            trackFillRatio: 1.0,
            trackColor: curRgb
          });

          if (c.thumbGlassMix > 0.01) {
            const worldThumbX = c.x - c.width * 0.5 + thumbLocalX;
            const capHalfW = 19.0 * c.thumbScale * this.dpr;
            const capHalfH = 13.0 * c.thumbScale * this.dpr;
            glassList.push({
              cx: worldThumbX * this.dpr,
              cy: c.y * this.dpr,
              hx: capHalfW,
              hy: capHalfH,
              radius: capHalfH,
              translucency: c.translucency,
              tintEnabled: 0.0,
              tintOpacity: 0.0,
              tintColor: [1.0, 1.0, 1.0],
              adhesion: 0,
              tx: worldThumbX * this.dpr,
              ty: c.y * this.dpr,
              glow: 0.6 * c.thumbGlassMix,
              layer: c.layerId,
              skewX: 0,
              skewY: 0,
              glassActive: c.thumbGlassMix,
              solidTrack: 0,
              trackFillRatio: 0,
              trackColor: curRgb
            });
          }
          continue;
        }

        if (c.glassActive > 0.01) {
          const rgb = hexToRgb01(c.tintColor);
          glassList.push({
            cx: c.x * this.dpr,
            cy: c.y * this.dpr,
            hx: c.width * 0.5 * c.scale * this.dpr,
            hy: c.height * 0.5 * c.scale * this.dpr,
            radius: c.borderRadius * c.scale * this.dpr,
            translucency: c.translucency,
            tintEnabled: c.tint ? 1.0 : 0.0,
            tintOpacity: c.tintOpacity,
            tintColor: rgb,
            adhesion: c.adhesion ? 1.0 : 0.0,
            tx: worldTouchX * this.dpr,
            ty: worldTouchY * this.dpr,
            glow: c.glowIntensity,
            layer: c.layerId,
            skewX: c.skewX * this.dpr,
            skewY: c.skewY * this.dpr,
            glassActive: c.glassActive,
            solidTrack: 0.0,
            trackFillRatio: 0.0,
            trackColor: rgb
          });

          if (c.type === "toolbar") {
            const npx = stepPos(c.pillOffsetX, c.vPillOffsetX, c.targetPillOffsetX, 240.0, 17.0, rawDt);
            c.vPillOffsetX = stepVel(c.pillOffsetX, c.vPillOffsetX, c.targetPillOffsetX, 240.0, 17.0, rawDt);
            c.pillOffsetX = npx;

            const npw = stepPos(c.pillHalfW, c.vPillHalfW, c.targetPillHalfW, 230.0, 17.0, rawDt);
            c.vPillHalfW = stepVel(c.pillHalfW, c.vPillHalfW, c.targetPillHalfW, 230.0, 17.0, rawDt);
            c.pillHalfW = npw;

            const nps = stepPos(c.pillScale, c.vPillScale, c.targetPillScale, 250.0, 14.0, rawDt);
            c.vPillScale = stepVel(c.pillScale, c.vPillScale, c.targetPillScale, 250.0, 14.0, rawDt);
            c.pillScale = nps;

            const npg = stepPos(c.pillGlow, c.vPillGlow, c.targetPillGlow, 190.0, 16.0, rawDt);
            c.vPillGlow = stepVel(c.pillGlow, c.vPillGlow, c.targetPillGlow, 190.0, 16.0, rawDt);
            c.pillGlow = Math.max(0, npg);

            totalEnergy += Math.abs(c.vPillOffsetX) + Math.abs(c.vPillHalfW) + Math.abs(c.vPillScale);

            const stretchW = Math.min(12.0, Math.abs(c.vPillOffsetX) * 0.035);
            const pillWorldX = c.x + c.pillOffsetX * c.scale;
            const pillHX = (c.pillHalfW * c.pillScale + stretchW) * c.scale * this.dpr;
            const pillHY = (c.pillHalfH * c.pillScale) * c.scale * this.dpr;

            glassList.push({
              cx: pillWorldX * this.dpr,
              cy: c.y * this.dpr,
              hx: pillHX,
              hy: pillHY,
              radius: pillHY,
              translucency: Math.max(15, c.translucency * 0.25),
              tintEnabled: 1.0,
              tintOpacity: 0.10,
              tintColor: [0.70, 0.90, 1.0],
              adhesion: 0.0,
              tx: pillWorldX * this.dpr,
              ty: c.y * this.dpr,
              glow: 0.45 + c.pillGlow,
              layer: c.layerId + 0.1,
              skewX: (c.vPillOffsetX * 0.012) * this.dpr,
              skewY: 0,
              glassActive: 1.0,
              solidTrack: 0.0,
              trackFillRatio: 0.0,
              trackColor: rgb
            });
          }
        }
      }

      this.updateDOMOcclusionMasks();

      // 仅对激活的玻璃图元按 layer 升序排列，轨道图元固定排在 [0 .. trackCount)
      glassList.sort((a, b) => a.layer - b.layer);

      let gpuCount = 0;
      for (let i = 0; i < trackList.length && gpuCount < this.maxControls; i++) {
        this.packGPUPrimitive(this.gpuData, gpuCount++, trackList[i]);
      }
      const trackCount = gpuCount;
      for (let i = 0; i < glassList.length && gpuCount < this.maxControls; i++) {
        this.packGPUPrimitive(this.gpuData, gpuCount++, glassList[i]);
      }

      if (hasWebGPU) {
        this.globalsArr[0] = this.canvas.width;
        this.globalsArr[1] = this.canvas.height;
        this.globalsArr[2] = elapsed;
        this.globalsU32[3] = gpuCount;
        this.globalsArr[4] = this.gyroX;
        this.globalsArr[5] = this.gyroY;
        this.globalsArr[6] = this.gyroBoost;
        this.globalsArr[7] = this.modalDim;
        this.globalsU32[8] = trackCount;

        this.device.queue.writeBuffer(this.uniformBuffer, 0, this.globalsArr);
        if (gpuCount > 0) {
          this.device.queue.writeBuffer(this.storageBuffer, 0, this.gpuData, 0, gpuCount * 24);
        }

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
          colorAttachments: [{
            view: this.context.getCurrentTexture().createView(),
            loadOp: "clear",
            clearValue: { r: 0.03, g: 0.04, b: 0.08, a: 1.0 },
            storeOp: "store"
          }]
        });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.draw(6, 1, 0, 0);
        pass.end();
        this.device.queue.submit([encoder.finish()]);
      } else if (this.fallbackCtx) {
        this.fallbackCtx.drawImage(this.bgCanvas, 0, 0);
      }

      // 若系统动能已收敛且背景非动态动画模式，则自动休眠释放 CPU/GPU
      const isStaticBg = this.backgroundMode === "element" && (!this.customBgSource || this.customBgSource.tagName !== "VIDEO");
      if (isStaticBg && totalEnergy < 0.002) {
        this.isSleeping = true;
        this.rafId = 0;
        return;
      }

      this.rafId = requestAnimationFrame((t) => this.frameLoop(t));
    }

    packGPUPrimitive(buf, idx, p) {
      const o = idx * 24;
      buf[o + 0] = p.cx;
      buf[o + 1] = p.cy;
      buf[o + 2] = p.hx;
      buf[o + 3] = p.hy;
      buf[o + 4] = p.radius;
      buf[o + 5] = p.translucency;
      buf[o + 6] = p.tintEnabled;
      buf[o + 7] = p.tintOpacity;
      buf[o + 8] = p.tintColor[0];
      buf[o + 9] = p.tintColor[1];
      buf[o + 10] = p.tintColor[2];
      buf[o + 11] = p.adhesion;
      buf[o + 12] = p.tx;
      buf[o + 13] = p.ty;
      buf[o + 14] = p.glow;
      buf[o + 15] = p.layer;
      buf[o + 16] = p.skewX;
      buf[o + 17] = p.skewY;
      buf[o + 18] = p.glassActive;
      buf[o + 19] = p.solidTrack;
      buf[o + 20] = p.trackColor[0];
      buf[o + 21] = p.trackColor[1];
      buf[o + 22] = p.trackColor[2];
      buf[o + 23] = p.trackFillRatio;
    }

    /**
     * 快速挂载全套 10 类组件的示例工作台 (可通过 ui.clear() 一键清空)
     */
    mountShowcase() {
      const cx = (this.stageWidth || window.innerWidth) * 0.5;
      const cy = (this.stageHeight || window.innerHeight) * 0.5;

      this.text({
        label: "✦ LiquidGlass SDK v2.0",
        x: 145, y: 52, width: 215, height: 40,
        tintColor: "#5ac8fa", tintOpacity: 0.12, translucency: 80, layerId: 2
      });

      this.toolbar({
        options: ["组件概览", "折射引擎", "光谱色散", "API文档"],
        selectedIndex: 0,
        x: 485, y: 52, width: 400, height: 48,
        tintColor: "#5e5ce6", tintOpacity: 0.12, translucency: 100, adhesion: false, layerId: 2
      });

      const alertTrigger = this.button({
        label: "🔔 弹出居中 Alert",
        x: 810, y: 52, width: 185, height: 44,
        tintColor: "#ff9f0a", tintOpacity: 0.18, translucency: 90, adhesion: false, layerId: 2,
        hiddenByMorph: true
      });

      const modalAlert = this.alert({
        label: "LiquidGlass 组件库已就绪",
        description: "可通过 window.LiquidGlass.init() 或 ui.button() / ui.slider() 等 API 动态创建和操控任意控件。",
        x: cx, y: cy, width: 360, height: 195,
        tintColor: "#38b6ff", tintOpacity: 0.12, borderRadius: 30, translucency: 130, adhesion: false, layerId: 10,
        morphTargetId: alertTrigger.id
      });
      alertTrigger.morphTargetId = modalAlert.id;

      this.button({
        label: "主按钮 (拖拽粘连)",
        x: 155, y: 150, width: 195, height: 50,
        tintColor: "#00c8ff", tintOpacity: 0.16, borderRadius: 25, translucency: 75, layerId: 1
      });

      this.button({
        label: "异色按钮 (渐变融合)",
        x: 375, y: 150, width: 195, height: 50,
        tintColor: "#ff3b9a", tintOpacity: 0.18, borderRadius: 16, translucency: 160, layerId: 1
      });

      this.div({
        label: "高折射色散容器 (ui.div)",
        description: "支持 ctrl.setProps({ translucency, tintColor, width, height }) 实时响应式更新。拖拽可观察强烈透镜扭曲与 5 波段光谱色散。",
        x: 265, y: 285, width: 415, height: 140,
        tintColor: "#bf5af2", tintOpacity: 0.10, borderRadius: 28, translucency: 85, layerId: 1
      });

      this.editor({
        description: "富文本编辑器 (ui.editor)",
        label: "// 直接在此输入代码或文本\nconst btn = ui.button({ label: 'Hello', x: 300, y: 200 });\nbtn.on('click', () => modalAlert.show());",
        x: 265, y: 460, width: 415, height: 145,
        tintColor: "#64d2ff", tintOpacity: 0.10, borderRadius: 26, translucency: 105, layerId: 1
      });

      const dropdown = this.dropdown({
        label: "预设: 强折射棱镜色散模式",
        x: 720, y: 245, width: 340, height: 48,
        tintColor: "#30d5c8", tintOpacity: 0.14, borderRadius: 22, translucency: 95, adhesion: false, layerId: 4
      });

      const menu = this.menu({
        options: [
          "高透晶体 (通透性 20)",
          "强折射棱镜色散模式",
          "超长文本选项：80%深色圆角遮罩随文字宽度自适应变形",
          "深度磨砂背景 (通透性 900)"
        ],
        selectedIndex: 1,
        x: 720, y: 420, width: 350, height: 185,
        tintColor: "#38b6ff", tintOpacity: 0.12, borderRadius: 26, translucency: 120, adhesion: false, layerId: 5,
        onChange: (idx) => {
          const map = [20, 120, 260, 900];
          menu.setProps({ translucency: map[idx] || 120 });
        }
      });
      dropdown.morphTargetId = menu.id;

      this.slider({
        value: 0.64,
        x: 680, y: 150, width: 265, height: 44,
        tintColor: "#0a84ff", translucency: 25, layerId: 3,
        onChange: (val) => {
          menu.setProps({ translucency: Math.round(val * 1024) });
        }
      });

      this.switch({
        checked: true,
        x: 880, y: 150, width: 84, height: 40,
        tintColor: "#34c759", translucency: 25, layerId: 3
      });

      return this;
    }
  }

  /* ============================================================================
   * 7. 暴露全局 SDK 接口 (window.LiquidGlass & window.LiquidGlassUI)
   * ============================================================================ */
  const LiquidGlass = {
    Engine: LiquidGlassEngine,
    Control: LiquidGlassControl,
    defaultInstance: null,

    async init(options = {}) {
      const engine = new LiquidGlassEngine(options);
      await engine.init();
      if (!this.defaultInstance) this.defaultInstance = engine;
      if (options.autoShowcase || (options.autoShowcase !== false && engine.controls.length === 0 && !window.LIQUID_GLASS_DISABLE_AUTO_DEMO)) {
        engine.mountShowcase();
      }
      return engine;
    },

    create(type, props) {
      if (!this.defaultInstance) throw new Error("请先调用 await LiquidGlass.init()");
      return this.defaultInstance.create(type, props);
    },

    clear() {
      if (this.defaultInstance) this.defaultInstance.clear();
    }
  };

  window.LiquidGlass = LiquidGlass;
  window.LiquidGlassUI = LiquidGlass;

  // 若页面未配置手动初始化标记，则自动启动默认实例
  if (!window.LIQUID_GLASS_MANUAL_INIT) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", () => LiquidGlass.init());
    } else {
      LiquidGlass.init();
    }
  }
})();
