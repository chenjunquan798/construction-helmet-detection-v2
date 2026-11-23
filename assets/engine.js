/* 工地入口安全帽检测系统 - 推理 / 决策 / 复述引擎
 *
 * 默认使用 SimulationSource：确定性可复现的推理仿真，保证系统开箱即跑、结果可复述。
 * 接入真实 YOLOv8 时替换为 RemoteYoloSource（见 inference/yolov8_service.py），
 * 两者输出同一 JSON schema，上层 UI 与决策链无需改动。
 */
(function (global) {
  'use strict';

  var D = global.SITE_DATA;
  var BASE_TS = new Date(2026, 10, 23, 10, 20, 0).getTime();

  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function r3(v) { return Math.round(v * 1000) / 1000; }
  function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }
  function jit(rand, base, spread, lo, hi) {
    return r3(clamp(base + (rand() * 2 - 1) * spread, lo === undefined ? 0 : lo, hi === undefined ? 1 : hi));
  }
  function pad(n) { return n < 10 ? '0' + n : '' + n; }
  function tsOf(seed) {
    var d = new Date(BASE_TS + (seed % 3600) * 1000);
    return pad(d.getHours()) + ':' + pad(d.getMinutes()) + ':' + pad(d.getSeconds());
  }

  /* ---------- 场景解析 ---------- */
  function resolveScene(scene, staff) {
    var s = {};
    for (var k in scene) if (Object.prototype.hasOwnProperty.call(scene, k)) s[k] = scene[k];
    s.color = scene.color === null || scene.color === undefined ? staff.color : scene.color;
    s.authorized = staff.color;
    /* 动态真值：同一场景对不同授权帽色的员工，合规结论可能不同 */
    s.expectedAllow = !!(
      s.helmetPresent && s.wear === 'correct' && s.damage === 'intact' &&
      s.color === staff.color && s.live && s.faceScore >= 0.70
    );
    return s;
  }

  /* ---------- 推理源：仿真 ---------- */
  var SimulationSource = {
    name: '仿真推理',
    mode: 'simulation',
    infer: function (input) {
      var s = input.scene, staff = input.staff, rand = mulberry32(input.seed || 1);
      var onHead = s.helmetPresent && s.wear !== 'handheld';
      var helmetConf = s.helmetPresent ? jit(rand, 0.965, 0.028, 0.85, 0.995) : 0;

      var boxes = {
        head:   { x: 286, y: 132, w: 88,  h: 96 },
        face:   { x: 300, y: 152, w: 60,  h: 62 },
        strap:  { x: 296, y: 196, w: 68,  h: 44 }
      };
      boxes.helmet = onHead
        ? { x: 278, y: 110, w: 104, h: 58 }
        : { x: 432, y: 244, w: 92,  h: 54 };

      var items = [
        { cls: 'head',   label: '头部',     box: boxes.head,   conf: jit(rand, 0.96, 0.02) },
        { cls: 'face',   label: '人脸',     box: boxes.face,   conf: jit(rand, 0.94, 0.03) }
      ];
      if (s.helmetPresent) {
        items.push({ cls: 'helmet', label: '安全帽', box: boxes.helmet, conf: helmetConf });
        items.push({ cls: 'strap',  label: '下颚带', box: boxes.strap,  conf: s.wear === 'unstrapped' ? jit(rand, 0.42, 0.08) : jit(rand, 0.91, 0.04) });
      }

      /* 颜色分支：HSV 主色 + YOLOv8-cls 复核 */
      var cInfo = D.COLORS[s.color];
      var clsProbs = {}, rest = 1;
      var mainP = jit(rand, 0.93, 0.045, 0.82, 0.99);
      var others = Object.keys(D.COLORS).filter(function (k) { return k !== s.color; });
      var shares = others.map(function () { return rand() + 0.15; });
      var sum = shares.reduce(function (a, b) { return a + b; }, 0);
      others.forEach(function (k, i) { clsProbs[k] = r3((1 - mainP) * shares[i] / sum); });
      clsProbs[s.color] = mainP;
      rest = Object.keys(clsProbs).reduce(function (a, k) { return a + clsProbs[k]; }, 0);

      /* 破损分支：盔体裁剪后分类 */
      var dmgBase = { intact: 0.06, crack: 0.92, dent: 0.88, chip: 0.85, aging: 0.79 }[s.damage] || 0.06;
      var dmgProb = jit(rand, dmgBase, 0.045, 0.01, 0.99);
      var dmgBoxes = [];
      if (s.damage !== 'intact') {
        dmgBoxes.push({ x: 300, y: 120, w: 60, h: 34, label: D.DAMAGE_LABELS[s.damage] });
      }

      /* 佩戴分支：YOLOv8-pose 关键点 + 几何规则 */
      var wc = {
        coverage: true, front: true, offset: true, strap: true, orientation: true
      };
      if (s.wear === 'unstrapped') { wc.strap = false; }
      if (s.wear === 'backward')  { wc.orientation = false; wc.front = false; }
      if (s.wear === 'handheld')  { wc.coverage = false; wc.front = false; wc.offset = false; wc.strap = false; }
      if (s.wear === 'tilted')    { wc.offset = false; wc.coverage = false; }
      var wearCorrect = Object.keys(wc).every(function (k) { return wc[k]; });
      var wearConf = wearCorrect ? jit(rand, 0.95, 0.03) : jit(rand, 0.28, 0.09, 0.12, 0.45);

      var faceScore = jit(rand, s.faceScore, 0.012, 0.2, 0.99);

      return {
        source: 'simulation', seed: input.seed, ts: tsOf(input.seed || 1), channel: input.channel,
        detection: {
          items: items, boxes: boxes, helmetConf: helmetConf,
          helmetOnHead: onHead, count: items.length,
          model: 'YOLOv8s-seg / detect'
        },
        color: {
          hsvHue: Math.round(cInfo.hue + (rand() * 8 - 4)),
          clsProbs: clsProbs, probSum: r3(rest),
          predicted: s.color, authorized: s.authorized,
          conf: mainP, model: 'HSV + YOLOv8s-cls'
        },
        damage: {
          prob: dmgProb, cls: s.damage === 'intact' ? '完好' : D.DAMAGE_LABELS[s.damage],
          raw: s.damage, boxes: dmgBoxes, model: 'YOLOv8s-cls / OBB (crop 512)'
        },
        wear: {
          checks: wc, correct: wearCorrect, conf: wearConf,
          mode: s.wear, label: D.WEAR_LABELS[s.wear],
          keypoints: [
            { name: '左耳', x: 292, y: 186 }, { name: '右耳', x: 368, y: 186 },
            { name: '鼻',   x: 330, y: 186 }, { name: '头顶', x: 330, y: 128 },
            { name: '下巴', x: 330, y: 224 }
          ],
          model: 'YOLOv8s-pose + geometry'
        },
        face: {
          score: faceScore, live: s.live, matched: s.live && faceScore >= 0.55,
          staffId: staff.id, name: staff.name, role: staff.role, dept: staff.dept,
          model: 'ArcFace 1:N + 活体检测'
        }
      };
    }
  };

  /* ---------- 推理源：真实 YOLOv8 服务（可选，支持上传图片） ---------- */
  function RemoteYoloSource(url) {
    this.url = url; this.name = '真实 YOLOv8 服务'; this.mode = 'remote';
  }
  RemoteYoloSource.prototype.infer = function (input) {
    var self = this;
    var payload = {
      frame_id: 'frame_' + input.seed,
      channel: input.channel,
      staff_id: input.staff.id,
      face_name: input.staff.name,
      face_score: input.scene.faceScore,
      face_live: input.scene.live
    };
    if (input.image_b64) payload.image_b64 = input.image_b64;
    return fetch(this.url + '/infer', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (r) {
      if (!r.ok) throw new Error('推理服务返回 ' + r.status);
      return r.json();
    }).then(function (raw) {
      var out = normalizeRemote(raw, input);
      self.last = out; return out;
    });
  };

  /* 把服务返回值补齐成前端统一 schema，并把原图坐标映射到 720×400 画布 */
  function normalizeRemote(raw, input) {
    var staff = input.staff, view = input.view || { scale: 1, ox: 0, oy: 0 };
    function mapBox(b) {
      if (!b) return null;
      return {
        x: Math.round(b.x * view.scale + view.ox),
        y: Math.round(b.y * view.scale + view.oy),
        w: Math.round(b.w * view.scale),
        h: Math.round(b.h * view.scale)
      };
    }
    raw.source = 'yolov8'; raw.seed = input.seed; raw.ts = tsOf(input.seed);
    raw.channel = input.channel; raw.view = view;
    raw.degraded_branches = raw.degraded_branches || [];

    var det = raw.detection || {};
    det.items = (det.items || []).map(function (it) {
      var b = mapBox(it.box) || { x: 0, y: 0, w: 0, h: 0 };
      return { cls: it.cls, label: it.label || it.cls, box: b, conf: Number(it.conf) || 0 };
    });
    det.boxes = {};
    det.items.forEach(function (it) { det.boxes[it.cls] = it.box; });
    det.helmetConf = Number(det.helmetConf) || 0;
    det.helmetOnHead = !!det.helmetOnHead;
    det.count = det.items.length;
    det.model = det.model || 'YOLOv8s-seg / detect';
    raw.detection = det;

    var c = raw.color || {};
    c.authorized = staff.color;
    c.predicted = D.COLORS[c.predicted] ? c.predicted : staff.color;
    c.clsProbs = c.clsProbs || {};
    Object.keys(D.COLORS).forEach(function (k) { c.clsProbs[k] = Number(c.clsProbs[k]) || 0.01; });
    c.conf = Number(c.conf) || c.clsProbs[c.predicted] || 0.5;
    c.hsvHue = Number(c.hsvHue) || 0;
    c.model = c.model || 'YOLOv8s-cls';
    raw.color = c;

    var d = raw.damage || {};
    d.prob = Number(d.prob) || 0;
    d.cls = d.cls || '完好';
    d.boxes = (d.boxes || []).map(function (b) {
      var m = mapBox(b); return m ? { x: m.x, y: m.y, w: m.w, h: m.h, label: b.label || d.cls } : null;
    }).filter(function (b) { return b && b.w > 0 && b.h > 0; });
    d.model = d.model || 'YOLOv8s-cls / OBB (crop 512)';
    raw.damage = d;

    var w = raw.wear || {};
    w.checks = w.checks || { coverage: true, front: true, offset: true, strap: true, orientation: true };
    w.correct = Object.keys(w.checks).every(function (k) { return w.checks[k]; });
    w.conf = Number(w.conf) || (w.correct ? 0.9 : 0.3);
    w.label = w.label || (w.correct ? '规范佩戴' : '待复核');
    w.keypoints = (w.keypoints || []).map(function (k) {
      return { name: k.name, x: Math.round(k.x * view.scale + view.ox), y: Math.round(k.y * view.scale + view.oy) };
    });
    w.model = w.model || 'YOLOv8s-pose + geometry';
    raw.wear = w;

    var f = raw.face || {};
    f.score = Number(f.score) || input.scene.faceScore;
    f.live = f.live !== undefined ? !!f.live : !!input.scene.live;
    f.matched = f.live && f.score >= 0.55;
    f.name = staff.name; f.staffId = staff.id; f.role = staff.role; f.dept = staff.dept;
    f.model = f.model || '既有刷脸系统（1:N + 活体）';
    raw.face = f;
    return raw;
  }

  /* 图片模式：把「浏览器端像素分析 + 操作员复核」转换成统一 result */
  function buildResult(cfg) {
    var resolved = resolveScene(cfg.scene, cfg.staff);
    var r = SimulationSource.infer({ scene: resolved, staff: cfg.staff, seed: cfg.seed, channel: cfg.channel });
    r.source = cfg.yolo ? 'yolov8' : 'image-analysis';
    r.origin = cfg.origin || 'image';
    r.image = cfg.image || null;
    r.view = cfg.view || null;
    r.analysis = cfg.analysis || null;
    r.autoColor = !!cfg.autoColor;
    r.yolo = !!cfg.yolo;

    if (cfg.boxes && cfg.boxes.length) {
      r.detection.items = cfg.boxes;
      r.detection.count = cfg.boxes.length;
      r.detection.boxes = {};
      cfg.boxes.forEach(function (it) { r.detection.boxes[it.cls] = it.box; });
      var hb = cfg.boxes.filter(function (it) { return it.cls === 'helmet'; })[0];
      r.detection.helmetConf = hb ? hb.conf : 0;
    }
    r.detection.helmetOnHead = !!resolved.helmetPresent && resolved.wear !== 'handheld' &&
      (cfg.helmetOnHead === undefined ? true : !!cfg.helmetOnHead);
    r.detection.model = cfg.yolo ? 'YOLOv8s-seg / detect' : '浏览器端候选区 + 操作员复核';

    if (cfg.analysis) {
      r.color.hsvHue = cfg.analysis.hsvHue;
      r.color.clsProbs = cfg.analysis.clsProbs;
      r.color.conf = cfg.analysis.conf;
      r.color.model = cfg.analysis.model;
    }
    r.color.predicted = resolved.color;
    r.color.authorized = cfg.staff.color;

    if (!cfg.yolo) {
      r.damage.prob = resolved.damage === 'intact' ? 0.05 : 0.90;
      r.damage.cls = resolved.damage === 'intact' ? '完好' : D.DAMAGE_LABELS[resolved.damage];
      r.wear.conf = r.wear.correct ? 0.95 : 0.25;
      r.face.score = resolved.faceScore;
      r.face.live = resolved.live;
      r.face.matched = resolved.live && resolved.faceScore >= 0.55;
      r.damage.model = '操作员复核（图片模式）';
      r.wear.model = '关键点规则 + 操作员复核（图片模式）';
      r.face.model = '既有刷脸系统 / 操作员录入';
    }
    r.damage.boxes = cfg.damageBoxes || [];
    return r;
  }

  /* ---------- 决策引擎 ---------- */
  var DEFAULT_ENABLED = { face: true, helmet: true, wear: true, damage: true, color: true };
  var STRATEGIES = {
    full: { id: 'full', label: '完整四任务方案', enabled: DEFAULT_ENABLED },
    detectOnly: {
      id: 'detectOnly', label: '仅检测「有无安全帽」传统方案',
      enabled: { face: true, helmet: true, wear: false, damage: false, color: false }
    }
  };

  function decide(result, options) {
    var opts = options || {};
    var th = opts.thresholds || D.DEFAULT_THRESHOLDS;
    var en = Object.assign({}, DEFAULT_ENABLED, opts.enabled || {});
    var checks = [];

    function push(id, pass, detail, value, threshold, skipped) {
      var rule = D.RULES.filter(function (r) { return r.id === id; })[0];
      checks.push({
        id: id, order: rule.order, label: rule.label, hint: rule.hint,
        pass: !!pass, skipped: !!skipped, detail: detail,
        value: value, threshold: threshold
      });
    }

    var f = result.face;
    push('face', !en.face || (f.live && f.score >= th.faceScore),
      en.face
        ? '活体 ' + (f.live ? '通过' : '失败') + '，1:N 相似度 ' + f.score.toFixed(3) + '（阈值 ' + th.faceScore.toFixed(2) + '）'
        : '策略已关闭该规则，跳过校验',
      f.score, th.faceScore, !en.face);

    var det = result.detection;
    var helmetPass = det.helmetConf >= th.detConf && det.helmetOnHead;
    push('helmet', !en.helmet || helmetPass,
      en.helmet
        ? '盔体置信度 ' + det.helmetConf.toFixed(3) + '，' + (det.helmetOnHead ? '位于头顶区域' : '未位于头顶区域')
        : '策略已关闭该规则，跳过校验',
      det.helmetConf, th.detConf, !en.helmet);

    var w = result.wear;
    push('wear', !en.wear || (w.correct && w.conf >= th.wearConf),
      en.wear
        ? '佩戴状态「' + w.label + '」，几何校验 ' + (w.correct ? '全部通过' : '存在未通过项') + '，置信度 ' + w.conf.toFixed(3)
        : '策略已关闭该规则，跳过校验',
      w.conf, th.wearConf, !en.wear);

    var d = result.damage;
    push('damage', !en.damage || d.prob <= th.damageProb,
      en.damage
        ? '盔体分类「' + d.cls + '」，破损概率 ' + d.prob.toFixed(3) + '（阈值 ' + th.damageProb.toFixed(2) + '）'
        : '策略已关闭该规则，跳过校验',
      d.prob, th.damageProb, !en.damage);

    var c = result.color;
    push('color', !en.color || (c.predicted === c.authorized && c.conf >= th.colorConf),
      en.color
        ? '佩戴 ' + D.COLORS[c.predicted].name + '（' + D.COLORS[c.predicted].grade + '），授权 ' + D.COLORS[c.authorized].name + '，分类置信度 ' + c.conf.toFixed(3)
        : '策略已关闭该规则，跳过校验',
      c.conf, th.colorConf, !en.color);

    checks.sort(function (a, b) { return a.order - b.order; });
    var failed = checks.filter(function (x) { return !x.pass; });
    var allow = failed.length === 0;
    var code = allow ? 'ALLOW' : 'DENY_' + failed[0].id.toUpperCase();
    return {
      allow: allow, code: code, checks: checks, failed: failed,
      summary: allow
        ? '五项校验全部通过，闸机放行'
        : '第 ' + failed[0].order + ' 项「' + failed[0].label + '」未通过，闸机保持关闭'
    };
  }

  var CODE_TEXT = {
    ALLOW: '放行', DENY_FACE: '人脸未通过', DENY_HELMET: '未佩戴安全帽',
    DENY_WEAR: '佩戴不规范', DENY_DAMAGE: '安全帽破损', DENY_COLOR: '帽色等级不符'
  };

  /* ---------- 复述生成 ---------- */
  function narrate(ctx) {
    var r = ctx.result, dec = ctx.decision, s = ctx.scene, st = ctx.staff;
    var c = r.color, d = r.damage, w = r.wear, det = r.detection, f = r.face;
    var parts = [];

    parts.push({
      title: '采集', icon: '01',
      text: r.ts + '，' + r.channel + ' 红外触发，正面相机（人脸 / 下颚带）与斜上方相机（盔体 / 破损）同步取帧，帧 ID frame_' + r.seed + '。员工 ' + st.name + '（工号 ' + st.id + '，' + st.dept + ' / ' + st.role + '）进入识别区。'
    });

    parts.push({
      title: 'YOLOv8 多任务推理', icon: '02',
      text: '① 检测（' + det.model + '）输出 ' + det.count + ' 个目标：' + det.items.map(function (i) { return i.label + ' ' + i.conf.toFixed(2); }).join('、') + '。' +
        '② 颜色分支（' + c.model + '）在盔体 mask 内取 HSV 主色 H≈' + c.hsvHue + '°，分类模型判定为' + D.COLORS[c.predicted].name + '（置信度 ' + c.conf.toFixed(3) + '），对应等级「' + D.COLORS[c.predicted].grade + '」。' +
        '③ 破损分支（' + d.model + '）对盔体裁剪图分类为「' + d.cls + '」，破损概率 ' + d.prob.toFixed(3) + '。' +
        '④ 姿态分支（' + w.model + '）取耳 / 鼻 / 头顶 / 下巴关键点，判定佩戴状态「' + w.label + '」，几何校验' + (w.correct ? '全部通过' : '未通过项：' + Object.keys(w.checks).filter(function (k) { return !w.checks[k]; }).join('、')) + '。'
    });

    var chain = dec.checks.map(function (x) {
      return x.order + '.' + x.label + (x.skipped ? '［策略关闭］' : x.pass ? ' ✔' : ' ✘');
    }).join('  →  ');
    parts.push({
      title: '决策引擎', icon: '03',
      text: '按「人脸 → 帽在 → 佩戴 → 破损 → 等级」顺序串联校验：' + chain + '。' + dec.summary + '。' + (dec.failed.length ? '失败明细：' + dec.failed.map(function (x) { return x.detail; }).join('；') : '各阈值均满足。')
    });

    var action = dec.allow
      ? '闸机开闸，通行记录写入台账。'
      : '闸机保持关闭，声光提示「' + (CODE_TEXT[dec.code] || '校验未通过') + '」，抓拍留证并推送安保复核。';
    parts.push({
      title: '结论与处置', icon: '04',
      text: '场景预设：' + s.label + '（' + s.desc + '）。真值期望：' + (s.expectedAllow ? '允许通行' : '禁止通行') + '；系统判定：' + (dec.allow ? '允许通行' : '禁止通行') + '。' +
        (dec.allow !== s.expectedAllow ? '⚠ 本次判定与真值不一致，需人工复核。' : '判定与真值一致。') + action
    });

    return {
      parts: parts,
      full: parts.map(function (p) { return '【' + p.title + '】' + p.text; }).join('\n')
    };
  }

  /* ---------- 批量仿真对比 ---------- */
  function batch(n, seed, thresholds) {
    var rand = mulberry32(seed || 20261123);
    var out = {};
    Object.keys(STRATEGIES).forEach(function (k) {
      out[k] = { allow: 0, deny: 0, leak: 0, falseReject: 0, label: STRATEGIES[k].label };
    });
    var samples = [];
    for (var i = 0; i < n; i++) {
      var scene = D.SCENES[Math.floor(rand() * D.SCENES.length)];
      var staff = D.STAFF[Math.floor(rand() * D.STAFF.length)];
      var rs = resolveScene(scene, staff);
      var result = SimulationSource.infer({ scene: rs, staff: staff, seed: seed + i, channel: D.CHANNELS[0] });
      var row = { scene: scene.label, staff: staff.name, truth: rs.expectedAllow };
      Object.keys(STRATEGIES).forEach(function (k) {
        var dec = decide(result, { thresholds: thresholds, enabled: STRATEGIES[k].enabled });
        row[k] = dec.allow;
        if (dec.allow) out[k].allow++; else out[k].deny++;
        if (dec.allow && !rs.expectedAllow) out[k].leak++;
        if (!dec.allow && rs.expectedAllow) out[k].falseReject++;
      });
      samples.push(row);
    }
    var A = out.full, B = out.detectOnly;
    var conclusion = '在 ' + n + ' 次仿真通行中，完整四任务方案漏放违规 ' + A.leak + ' 次，' +
      '仅检测有无帽的传统方案漏放 ' + B.leak + ' 次（多出 ' + (B.leak - A.leak) + ' 次）。' +
      '传统方案只能拦下「完全没戴帽」和「人脸未通过」，对破损、错色、未系带、反戴、手持等 ' +
      '佩戴了帽子但不合规的情况几乎不设防；完整方案误拒 ' + A.falseReject + ' 次。';
    return { n: n, seed: seed, stats: out, samples: samples, conclusion: conclusion };
  }

  /* ---------- 阈值灵敏度扫描 ---------- */
  function sweep(param, sceneId, staffId, from, to, steps) {
    var scene = D.SCENES.filter(function (s) { return s.id === sceneId; })[0];
    var staff = D.STAFF.filter(function (s) { return s.id === staffId; })[0];
    var rs = resolveScene(scene, staff);
    var result = SimulationSource.infer({ scene: rs, staff: staff, seed: 4242, channel: D.CHANNELS[0] });
    var pts = [];
    for (var i = 0; i < steps; i++) {
      var v = from + (to - from) * i / (steps - 1);
      var th = Object.assign({}, D.DEFAULT_THRESHOLDS);
      th[param] = r3(v);
      var dec = decide(result, { thresholds: th });
      pts.push({ x: r3(v), allow: dec.allow, code: dec.code });
    }
    return { param: param, scene: scene.label, result: result, points: pts };
  }

  global.HelmetEngine = {
    SimulationSource: SimulationSource,
    RemoteYoloSource: RemoteYoloSource,
    STRATEGIES: STRATEGIES,
    CODE_TEXT: CODE_TEXT,
    resolveScene: resolveScene,
    infer: function (input, source) { return (source || SimulationSource).infer(input); },
    buildResult: buildResult,
    normalizeRemote: normalizeRemote,
    decide: decide,
    narrate: narrate,
    batch: batch,
    sweep: sweep,
    tsOf: tsOf,
    BASE_TS: BASE_TS
  };
})(window);
