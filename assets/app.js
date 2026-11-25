/* 工地入口安全帽检测系统 - 界面与交互层 */
(function (global) {
  'use strict';

  var D = global.SITE_DATA, E = global.HelmetEngine;
  function $(id) { return document.getElementById(id); }
  function find(arr, key, val) { for (var i = 0; i < arr.length; i++) if (arr[i][key] === val) return arr[i]; return arr[0]; }
  function fx(v, n) { return Number(v).toFixed(n === undefined ? 2 : n); }

  var TH_META = {
    faceScore:  { label: '人脸相似度下限', min: 0.30, max: 0.99 },
    detConf:    { label: '盔体检出置信度下限', min: 0.20, max: 0.99 },
    wearConf:   { label: '佩戴判定置信度下限', min: 0.20, max: 0.99 },
    damageProb: { label: '破损概率上限（超过即拦截）', min: 0.05, max: 0.99 },
    colorConf:  { label: '颜色分类置信度下限', min: 0.20, max: 0.99 }
  };
  var BOX_COLOR = { head: '#2563a8', face: '#0e7c86', helmet: '#5b4bb7', strap: '#b3730f' };

  var state = {
    tab: 'monitor', view: 'dual', showBoxes: true, showKeypoints: true,
    channel: D.CHANNELS[0], staffId: D.STAFF[0].id, sceneId: 'S01',
    overrides: null,
    thresholds: Object.assign({}, D.DEFAULT_THRESHOLDS),
    enabled: Object.assign({}, E.STRATEGIES.full.enabled),
    seed: 1, source: null, history: [], last: null
  };

  /* ---------------- 初始化 ---------------- */
  function fillSelect(el, items, value) {
    el.innerHTML = items.map(function (it) {
      var v = typeof it === 'string' ? it : it.value;
      var t = typeof it === 'string' ? it : it.label;
      return '<option value="' + v + '">' + t + '</option>';
    }).join('');
    if (value !== undefined) el.value = value;
  }

  function init() {
    fillSelect($('selChannel'), D.CHANNELS, state.channel);
    fillSelect($('selStaff'), D.STAFF.map(function (s) {
      return { value: s.id, label: s.id + ' · ' + s.name + '（' + s.role + ' / ' + D.COLORS[s.color].name + '帽）' };
    }), state.staffId);
    var sceneOpts = D.SCENES.map(function (s) { return { value: s.id, label: s.id + ' · ' + s.label + '［' + s.tag + '］' }; });
    fillSelect($('selScene'), sceneOpts, state.sceneId);
    fillSelect($('selSceneA'), sceneOpts, 'S01');
    fillSelect($('selSceneB'), sceneOpts, 'S03');
    fillSelect($('selSweepScene'), sceneOpts, 'S03');
    fillSelect($('selCompareStaff'), D.STAFF.map(function (s) {
      return { value: s.id, label: s.name + ' · ' + s.role + '（授权' + D.COLORS[s.color].name + '）' };
    }), state.staffId);
    fillSelect($('selColor'), [{ value: 'auto', label: '跟随授权帽色（合规）' }].concat(
      Object.keys(D.COLORS).map(function (k) { return { value: k, label: D.COLORS[k].name + ' · ' + D.COLORS[k].grade }; })
    ), 'auto');
    fillSelect($('selDamage'), Object.keys(D.DAMAGE_LABELS).map(function (k) {
      return { value: k, label: D.DAMAGE_LABELS[k] };
    }), 'intact');
    fillSelect($('selWear'), Object.keys(D.WEAR_LABELS).map(function (k) {
      return { value: k, label: D.WEAR_LABELS[k] };
    }), 'correct');

    buildThresholds();
    buildEnabled();
    $('colorTable').innerHTML = Object.keys(D.COLORS).map(function (k) {
      var c = D.COLORS[k];
      return '<tr><td><span class="chip" style="background:' + c.hex + ';color:' + c.text + ';border:1px solid ' + c.stroke + '">' + c.name + '</span></td><td>' + c.grade + '</td><td>' + c.desc + '</td></tr>';
    }).join('');

    syncControls(find(D.SCENES, 'id', state.sceneId));
    bindEvents();
    startClock();
    runDetection({ silent: false });
    renderCompare();
    renderSweep();
    renderBatch(200, false);
  }

  function buildThresholds() {
    $('thresholdBox').innerHTML = Object.keys(TH_META).map(function (k) {
      var m = TH_META[k];
      return '<div class="field"><label>' + m.label + '</label><div class="range-row">' +
        '<input type="range" data-th="' + k + '" min="' + m.min + '" max="' + m.max + '" step="0.01" value="' + state.thresholds[k] + '">' +
        '<div class="range-val" id="val_' + k + '">' + fx(state.thresholds[k]) + '</div></div></div>';
    }).join('');
  }

  function buildEnabled() {
    $('enabledBox').innerHTML = D.RULES.map(function (r) {
      return '<label class="check"><input type="checkbox" data-en="' + r.id + '"' + (state.enabled[r.id] ? ' checked' : '') + '>' + r.label + '</label>';
    }).join('');
  }

  function syncControls(base) {
    $('selColor').value = (base.color === null || base.color === undefined) ? 'auto' : base.color;
    $('selDamage').value = base.damage;
    $('selWear').value = base.wear;
    $('rngFace').value = base.faceScore;
    $('faceScoreVal').textContent = fx(base.faceScore);
    $('chkLive').checked = base.live;
    $('chkHelmet').checked = base.helmetPresent;
    Object.keys(TH_META).forEach(function (k) {
      var el = $('thresholdBox').querySelector('[data-th="' + k + '"]');
      if (el) el.value = state.thresholds[k];
      var lab = $('val_' + k); if (lab) lab.textContent = fx(state.thresholds[k]);
    });
    D.RULES.forEach(function (r) {
      var el = $('enabledBox').querySelector('[data-en="' + r.id + '"]');
      if (el) el.checked = state.enabled[r.id];
    });
  }

  function readOverrides() {
    state.overrides = {
      color: $('selColor').value === 'auto' ? null : $('selColor').value,
      damage: $('selDamage').value, wear: $('selWear').value,
      faceScore: parseFloat($('rngFace').value),
      live: $('chkLive').checked, helmetPresent: $('chkHelmet').checked
    };
  }

  function currentScene() {
    var base = find(D.SCENES, 'id', state.sceneId);
    var s = Object.assign({}, base);
    if (state.overrides) {
      var o = state.overrides;
      s.color = o.color; s.damage = o.damage; s.wear = o.wear;
      s.faceScore = o.faceScore; s.live = o.live; s.helmetPresent = o.helmetPresent;
      s.label = base.label + '（人工微调）';
    }
    return { base: base, scene: s };
  }

  function getStaff() { return find(D.STAFF, 'id', state.staffId); }

  function snapshot() {
    return {
      channel: state.channel, staffId: state.staffId, sceneId: state.sceneId,
      overrides: state.overrides ? Object.assign({}, state.overrides) : null,
      thresholds: Object.assign({}, state.thresholds),
      enabled: Object.assign({}, state.enabled), seed: state.seed
    };
  }

  function applyConfig(cfg) {
    state.channel = cfg.channel; state.staffId = cfg.staffId; state.sceneId = cfg.sceneId;
    state.overrides = cfg.overrides ? Object.assign({}, cfg.overrides) : null;
    state.thresholds = Object.assign({}, cfg.thresholds);
    state.enabled = Object.assign({}, cfg.enabled);
    state.seed = cfg.seed;
    $('selChannel').value = cfg.channel;
    $('selStaff').value = cfg.staffId;
    $('selScene').value = cfg.sceneId;
    var base = find(D.SCENES, 'id', cfg.sceneId);
    if (cfg.overrides) {
      $('selColor').value = cfg.overrides.color === null ? 'auto' : cfg.overrides.color;
      $('selDamage').value = cfg.overrides.damage;
      $('selWear').value = cfg.overrides.wear;
      $('rngFace').value = cfg.overrides.faceScore;
      $('faceScoreVal').textContent = fx(cfg.overrides.faceScore);
      $('chkLive').checked = cfg.overrides.live;
      $('chkHelmet').checked = cfg.overrides.helmetPresent;
    } else { syncControls(base); }
    syncThresholdsOnly();
  }
  function syncThresholdsOnly() {
    Object.keys(TH_META).forEach(function (k) {
      var el = $('thresholdBox').querySelector('[data-th="' + k + '"]');
      if (el) el.value = state.thresholds[k];
      var lab = $('val_' + k); if (lab) lab.textContent = fx(state.thresholds[k]);
    });
    D.RULES.forEach(function (r) {
      var el = $('enabledBox').querySelector('[data-en="' + r.id + '"]');
      if (el) el.checked = state.enabled[r.id];
    });
  }

  /* ---------------- 主流程 ---------------- */
  function runDetection(opts) {
    /* 摄像头接管时，控制台参数变化应作用于当前实时帧，而不是重新跑一遍仿真 */
    if (global.CameraLab && global.CameraLab.isActive()) {
      global.CameraLab.tick();
      return Promise.resolve(state.last);
    }
    var cfg = snapshot();
    var staff = getStaff();
    var cs = currentScene();
    var resolved = E.resolveScene(cs.scene, staff);
    var input = { scene: resolved, staff: staff, seed: state.seed, channel: state.channel };
    var src = state.source || E.SimulationSource;
    return Promise.resolve(E.infer(input, src)).then(function (result) {
      var decision = E.decide(result, { thresholds: state.thresholds, enabled: state.enabled });
      var narration = E.narrate({ result: result, decision: decision, scene: resolved, staff: staff });
      var ctx = { result: result, decision: decision, narration: narration, scene: resolved, staff: staff, cfg: cfg };
      state.last = ctx;
      renderAll(ctx);
      if (!opts || !opts.silent) {
        state.history.unshift(Object.assign({}, ctx));
        if (state.history.length > 80) state.history.pop();
        renderHistory();
      }
      return ctx;
    }).catch(function (err) {
      toast('推理失败：' + err.message);
      console.error(err);
    });
  }

  function refreshFromLast() {
    if (global.CameraLab && global.CameraLab.isActive()) { global.CameraLab.refresh(); return; }
    if (!state.last) return;
    var c = state.last;
    c.decision = E.decide(c.result, { thresholds: state.thresholds, enabled: state.enabled });
    c.narration = E.narrate({ result: c.result, decision: c.decision, scene: c.scene, staff: c.staff });
    renderAll(c);
  }

  function renderAll(ctx) {
    renderScene(ctx.result, ctx.scene, ctx.decision);
    renderPipeline(ctx.result);
    renderDecision(ctx.decision);
    renderNarrative(ctx.narration);
    $('channelChip').textContent = ctx.result.channel;
    $('frameChip').textContent = 'frame_' + ctx.result.seed;
    $('pipelineSub').textContent = ctx.scene.label + ' · ' + ctx.result.ts;
    $('sourceChip').textContent = '推理源：' + (state.source ? '真实 YOLOv8 服务' : '仿真推理');
    $('modeChip').className = 'chip ' + (state.source ? 'green' : 'amber');
    $('modeChip').innerHTML = '<span class="dot"></span>' + (state.source ? '真实模型接入' : '演示仿真模式');
  }

  /* ---------------- 场景画面 ---------------- */
  function renderScene(result, scene, decision) {
    var dual = state.view === 'dual', topOnly = state.view === 'top';
    var dx = dual ? -85 : 30, clipW = dual ? 490 : 720;
    var col = D.COLORS[scene.color] || D.COLORS.yellow;
    var allow = decision.allow, code = decision.code;
    var onHead = scene.helmetPresent && scene.wear !== 'handheld';
    var cx = 330;

    var bg = '<rect x="0" y="0" width="720" height="400" fill="#f8fafc"/>' +
      '<rect x="0" y="292" width="720" height="108" fill="#e9eef4"/>' +
      '<path d="M0 292 H720" stroke="#cbd5e1" stroke-width="1"/>' +
      '<path d="M120 400 L300 292 M240 400 L340 292 M480 400 L380 292 M600 400 L420 292" stroke="#dbe3ed" stroke-width="1" fill="none"/>' +
      '<rect x="40" y="196" width="58" height="104" rx="6" fill="#cfd8e3" stroke="#94a3b8"/>' +
      '<rect x="622" y="196" width="58" height="104" rx="6" fill="#cfd8e3" stroke="#94a3b8"/>' +
      '<circle cx="69" cy="216" r="7" fill="' + (allow ? '#167a55' : '#c0392b') + '"/>' +
      '<circle cx="651" cy="216" r="7" fill="' + (allow ? '#167a55' : '#c0392b') + '"/>' +
      '<path d="M98 252 L146 252" stroke="' + (allow ? '#167a55' : '#c0392b') + '" stroke-width="6" stroke-linecap="round"/>' +
      '<path d="M622 252 L574 252" stroke="' + (allow ? '#167a55' : '#c0392b') + '" stroke-width="6" stroke-linecap="round"/>' +
      '<rect x="14" y="16" width="118" height="22" rx="5" fill="#fff" stroke="#e2e8f0"/>' +
      '<text x="22" y="31" font-size="11" fill="#64748b">正面相机 · CAM-01</text>';

    var person = '';
    person += '<path d="M272 300 L284 238 Q330 218 376 238 L388 300 Z" fill="#3f6fa8" stroke="#2c5488" stroke-width="1"/>';
    person += '<path d="M286 264 H374" stroke="#f5c518" stroke-width="6" opacity=".85"/>';
    person += '<rect x="316" y="210" width="28" height="28" rx="6" fill="#e8b78f"/>';
    person += '<ellipse cx="330" cy="178" rx="34" ry="42" fill="#f2c9a5" stroke="#d8a97f" stroke-width="1"/>';
    person += '<circle cx="318" cy="176" r="2.4" fill="#334155"/><circle cx="342" cy="176" r="2.4" fill="#334155"/>';
    person += '<path d="M330 180 L327 190 H333" stroke="#d8a97f" fill="none" stroke-width="1.2"/>';
    person += '<path d="M322 199 Q330 205 338 199" stroke="#b9755c" fill="none" stroke-width="1.4"/>';

    if (!scene.helmetPresent) {
      person += '<path d="M296 168 Q300 130 330 130 Q360 130 364 168 Q352 150 330 152 Q308 150 296 168 Z" fill="#3d3a36"/>';
    }

    var hat = '';
    if (scene.helmetPresent && onHead) {
      var rot = scene.wear === 'tilted' ? -18 : 0;
      hat += '<g transform="rotate(' + rot + ' 330 160)">';
      hat += '<path d="M280 166 A50 50 0 0 1 380 166 L380 170 L280 170 Z" fill="' + col.hex + '" stroke="' + col.stroke + '" stroke-width="1.5"/>';
      if (scene.wear === 'backward') {
        hat += '<ellipse cx="382" cy="166" rx="10" ry="26" fill="' + col.hex + '" stroke="' + col.stroke + '" stroke-width="1.2"/>';
      } else {
        hat += '<ellipse cx="330" cy="170" rx="56" ry="8" fill="' + col.hex + '" stroke="' + col.stroke + '" stroke-width="1.2"/>';
      }
      hat += '<path d="M294 148 Q330 118 366 148" stroke="rgba(255,255,255,.6)" stroke-width="3" fill="none"/>';
      hat += '<path d="M330 120 L330 166" stroke="' + col.stroke + '" stroke-width="1" opacity=".35"/>';
      hat += '</g>';
      if (scene.wear === 'backward') {
        hat += '<text x="392" y="128" font-size="11" fill="#c0392b" font-weight="600">帽檐朝后</text>';
      }
      if (scene.damage === 'crack') {
        hat += '<path d="M306 130 L316 146 L308 156 L324 164" stroke="#8b1a1a" stroke-width="2.2" fill="none" stroke-linecap="round"/>';
      } else if (scene.damage === 'dent') {
        hat += '<ellipse cx="340" cy="140" rx="19" ry="9" fill="#94a3b8" opacity=".65"/>';
      } else if (scene.damage === 'chip') {
        hat += '<path d="M370 168 L388 173 L372 177 Z" fill="#f8fafc"/><circle cx="378" cy="172" r="9" fill="none" stroke="#c0392b" stroke-width="1.5"/>';
      } else if (scene.damage === 'aging') {
        hat += '<circle cx="310" cy="148" r="2.4" fill="#94a3b8"/><circle cx="350" cy="140" r="2" fill="#94a3b8"/><circle cx="338" cy="156" r="2.2" fill="#94a3b8"/>';
      }
      if (scene.wear === 'unstrapped') {
        hat += '<path d="M286 170 Q276 214 282 246 M374 170 Q384 214 378 246" stroke="#b3730f" stroke-width="2" stroke-dasharray="5 4" fill="none"/>';
      } else {
        hat += '<path d="M286 170 Q300 214 330 224 Q360 214 374 170" stroke="#b3730f" stroke-width="2" fill="none"/>';
      }
    } else if (scene.helmetPresent && scene.wear === 'handheld') {
      person += '<path d="M296 168 Q300 130 330 130 Q360 130 364 168 Q352 150 330 152 Q308 150 296 168 Z" fill="#3d3a36"/>';
      hat += '<path d="M388 246 Q410 234 424 254" stroke="#e8b78f" stroke-width="9" fill="none" stroke-linecap="round"/>';
      hat += '<path d="M424 258 A36 36 0 0 1 496 258 L496 262 L424 262 Z" fill="' + col.hex + '" stroke="' + col.stroke + '" stroke-width="1.4"/>';
      hat += '<ellipse cx="460" cy="262" rx="42" ry="6" fill="' + col.hex + '" stroke="' + col.stroke + '" stroke-width="1.2"/>';
      hat += '<text x="418" y="288" font-size="11" fill="#c0392b" font-weight="600">手持，未戴在头上</text>';
    }
    person += hat;

    var overlay = '';
    if (state.showBoxes) {
      result.detection.items.forEach(function (it) {
        var b = it.box, c = BOX_COLOR[it.cls] || '#2563a8';
        var label = it.label + ' ' + fx(it.conf);
        overlay += '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" fill="none" stroke="' + c + '" stroke-width="1.6" stroke-dasharray="4 3"/>';
        overlay += '<rect x="' + b.x + '" y="' + (b.y - 15) + '" width="' + (label.length * 8.4 + 12) + '" height="15" rx="3" fill="' + c + '"/>';
        overlay += '<text x="' + (b.x + 5) + '" y="' + (b.y - 4) + '" font-size="10.5" fill="#fff">' + label + '</text>';
      });
      result.damage.boxes.forEach(function (b) {
        if (!b.w || !b.h) return;
        overlay += '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" fill="none" stroke="#c0392b" stroke-width="1.8"/>';
        overlay += '<rect x="' + b.x + '" y="' + (b.y - 15) + '" width="' + (b.label.length * 12 + 10) + '" height="15" rx="3" fill="#c0392b"/>';
        overlay += '<text x="' + (b.x + 5) + '" y="' + (b.y - 4) + '" font-size="10.5" fill="#fff">' + b.label + '</text>';
      });
    }
    if (state.showKeypoints) {
      result.wear.keypoints.forEach(function (k) {
        overlay += '<circle cx="' + k.x + '" cy="' + k.y + '" r="3.6" fill="#167a55" stroke="#fff" stroke-width="1.2"><title>' + k.name + '</title></circle>';
      });
    }

    var main = '';
    if (!topOnly) {
      main = '<g clip-path="url(#mainClip)"><g transform="translate(' + dx + ',0)">' + person + overlay + '</g></g>';
    }

    var topPanel = '';
    if (dual) topPanel = drawTop(502, 40, 204, 190, col, scene, false);
    else if (topOnly) topPanel = drawTop(210, 34, 300, 300, col, scene, true);

    var barW = dual ? 466 : 696;
    var bar = '<rect x="12" y="352" width="' + barW + '" height="36" rx="8" fill="' + (allow ? '#e6f4ec' : '#fceceb') + '" stroke="' + (allow ? '#bfe2ce' : '#f0c4c0') + '"/>' +
      '<circle cx="30" cy="370" r="6" fill="' + (allow ? '#167a55' : '#c0392b') + '"/>' +
      '<text x="44" y="375" font-size="13" font-weight="600" fill="' + (allow ? '#167a55' : '#c0392b') + '">' +
      (allow ? 'ALLOW · 闸机放行' : 'DENY · ' + code + ' 拦截') + '</text>' +
      '<text x="' + (12 + barW - 12) + '" y="374" text-anchor="end" font-size="10.5" fill="#64748b">' + result.channel + ' · ' + result.ts + '</text>';

    var svg = '<defs><clipPath id="mainClip"><rect x="0" y="0" width="' + clipW + '" height="400"/></clipPath></defs>' +
      bg + main + topPanel + bar +
      '<text x="708" y="392" text-anchor="end" font-size="10" fill="#94a3b8">YOLOv8 多任务叠加显示</text>';
    $('sceneSvg').innerHTML = svg;
  }

  function drawTop(x, y, w, h, col, scene, big) {
    var cx = x + w / 2, cy = y + h / 2 + 12, r = Math.min(w, h) * (big ? 0.30 : 0.31);
    var s = '<rect x="' + x + '" y="' + y + '" width="' + w + '" height="' + h + '" rx="10" fill="#fff" stroke="#e2e8f0"/>' +
      '<text x="' + (x + 12) + '" y="' + (y + 20) + '" font-size="11" fill="#64748b">斜上方相机 · 盔体裁剪</text>' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + col.hex + '" stroke="' + col.stroke + '" stroke-width="2"/>' +
      '<circle cx="' + cx + '" cy="' + cy + '" r="' + (r * 0.87) + '" fill="none" stroke="' + col.stroke + '" stroke-width="1" opacity=".45"/>' +
      '<path d="M' + (cx - r) + ' ' + cy + ' H' + (cx + r) + '" stroke="' + col.stroke + '" stroke-width="1" opacity=".35"/>' +
      '<path d="M' + (cx - r * 0.5) + ' ' + (cy - r * 0.86) + ' V' + (cy + r * 0.86) + '" stroke="' + col.stroke + '" stroke-width="1" opacity=".3"/>' +
      '<path d="M' + (cx + r * 0.5) + ' ' + (cy - r * 0.86) + ' V' + (cy + r * 0.86) + '" stroke="' + col.stroke + '" stroke-width="1" opacity=".3"/>';
    if (scene.damage === 'crack') {
      s += '<path d="M' + (cx - r * 0.6) + ' ' + (cy - r * 0.5) + ' L' + (cx - r * 0.1) + ' ' + (cy - r * 0.05) + ' L' + (cx - r * 0.35) + ' ' + (cy + r * 0.2) + ' L' + (cx + r * 0.3) + ' ' + (cy + r * 0.6) + '" stroke="#8b1a1a" stroke-width="2.2" fill="none" stroke-linecap="round"/>';
      s += '<circle cx="' + (cx - r * 0.1) + '" cy="' + (cy - r * 0.05) + '" r="' + (r * 0.42) + '" fill="none" stroke="#c0392b" stroke-width="1.4" stroke-dasharray="3 2"/>';
    } else if (scene.damage === 'dent') {
      s += '<ellipse cx="' + (cx + r * 0.2) + '" cy="' + (cy - r * 0.2) + '" rx="' + (r * 0.34) + '" ry="' + (r * 0.2) + '" fill="#94a3b8" opacity=".7"/>';
    } else if (scene.damage === 'chip') {
      s += '<path d="M' + (cx + r * 0.75) + ' ' + (cy - r * 0.4) + ' L' + (cx + r * 1.05) + ' ' + cy + ' L' + (cx + r * 0.75) + ' ' + (cy + r * 0.4) + ' Z" fill="#fff" stroke="#c0392b" stroke-width="1.4"/>';
    }
    var ok = scene.damage === 'intact';
    var tagW = ok ? 62 : 84;
    s += '<rect x="' + (x + 12) + '" y="' + (y + h - 30) + '" width="' + tagW + '" height="20" rx="5" fill="' + (ok ? '#e6f4ec' : '#fceceb') + '"/>' +
      '<text x="' + (x + 20) + '" y="' + (y + h - 16) + '" font-size="10.5" font-weight="600" fill="' + (ok ? '#167a55' : '#c0392b') + '">' +
      (ok ? '盔体完好' : '检出：' + D.DAMAGE_LABELS[scene.damage]) + '</text>';
    if (big) s += '<text x="' + (x + w / 2) + '" y="' + (y + h - 40) + '" text-anchor="middle" font-size="11" fill="#64748b">分类输入 512 × 512</text>';
    return s;
  }

  /* ---------------- 四任务输出 ---------------- */
  function renderPipeline(r) {
    var det = r.detection, c = r.color, d = r.damage, w = r.wear;
    var probs = Object.keys(D.COLORS).map(function (k) {
      return '<div><b style="background:' + D.COLORS[k].hex + ';opacity:' + (0.35 + c.clsProbs[k] * 0.65) + '"></b>' + D.COLORS[k].name + ' ' + fx(c.clsProbs[k], 2) + '</div>';
    }).join('');
    var cards = [
      { cls: 'b-blue', name: '① 目标检测', model: det.model, val: det.count + ' 目标',
        desc: det.items.map(function (i) { return i.label + ' ' + fx(i.conf); }).join(' · '), bar: det.helmetConf, extra: '' },
      { cls: 'b-amber', name: '② 颜色与等级', model: c.model, val: D.COLORS[c.predicted].name,
        desc: '等级「' + D.COLORS[c.predicted].grade + '」HSV H≈' + c.hsvHue + '° · 授权' + D.COLORS[c.authorized].name, bar: c.conf, extra: '<div class="probs">' + probs + '</div>' },
      { cls: 'b-red', name: '③ 破损检测', model: d.model, val: fx(d.prob, 3),
        desc: '盔体分类「' + d.cls + '」 · 破损概率（越高越危险）', bar: d.prob, extra: '' },
      { cls: 'b-green', name: '④ 佩戴判定', model: w.model, val: w.correct ? '规范' : '不规范',
        desc: w.label + ' · 几何项 ' + Object.keys(w.checks).filter(function (k) { return w.checks[k]; }).length + '/5 通过', bar: w.conf, extra: '' }
    ];
    $('pipeline').innerHTML = cards.map(function (t) {
      return '<div class="task ' + t.cls + '"><div class="t-head"><div><div class="t-name">' + t.name + '</div><div class="t-model">' + t.model + '</div></div></div>' +
        '<div class="t-val">' + t.val + '</div><div class="t-desc">' + t.desc + '</div>' +
        '<div class="bar"><span style="width:' + Math.round(t.bar * 100) + '%"></span></div>' + t.extra + '</div>';
    }).join('');
  }

  /* ---------------- 决策链 ---------------- */
  function renderDecision(dec) {
    var rows = dec.checks.map(function (x) {
      var mark = x.skipped ? '<span class="mark skip">—</span>' : (x.pass ? '<span class="mark ok">✔</span>' : '<span class="mark no">✘</span>');
      return '<div class="check-row"><div class="order">' + x.order + '</div><div>' +
        '<div class="cr-title">' + x.label + '</div>' +
        '<div class="cr-detail' + (x.skipped ? ' skip' : '') + '">' + x.detail + '</div></div>' + mark + '</div>';
    }).join('');
    var v = '<div class="verdict ' + (dec.allow ? 'allow' : 'deny') + '">' +
      '<div class="v-code">' + dec.code + ' · ' + (E.CODE_TEXT[dec.code] || '校验结果') + '</div>' +
      '<div class="v-text">' + (dec.allow ? '五项校验全部通过，闸机放行' : '闸机保持关闭，拒绝通行') + '</div>' +
      '<div class="v-sub">' + dec.summary + (dec.failed.length ? '；未通过 ' + dec.failed.length + ' 项' : '') + '</div></div>';
    $('decisionBox').innerHTML = rows + v;
  }

  function renderNarrative(n) {
    $('narrativeBox').innerHTML = n.parts.map(function (p) {
      return '<div class="narrative-part"><div class="np-head">' + p.icon + ' · ' + p.title + '</div><div class="np-text">' + p.text + '</div></div>';
    }).join('');
  }

  /* ---------------- 台账 ---------------- */
  function renderHistory() {
    $('historyCount').textContent = state.history.length + ' 条记录';
    if (!state.history.length) {
      $('historyBox').innerHTML = '<p class="cr-detail" style="text-align:center;padding:24px 0">暂无记录，点击「执行检测」生成通行记录。</p>';
      return;
    }
    var rows = state.history.map(function (h, i) {
      var consistent = h.decision.allow === h.scene.expectedAllow;
      return '<tr class="clickable" data-idx="' + i + '"><td>' + h.result.ts + '</td><td>' + h.result.channel + '</td>' +
        '<td>' + h.staff.name + '<span class="cr-detail">' + h.staff.id + '</span></td>' +
        '<td>' + h.scene.label + '</td><td>' + h.decision.code + '</td>' +
        '<td><span class="tag ' + (h.decision.allow ? 'ok' : 'no') + '">' + (h.decision.allow ? '放行' : '拦截') + '</span></td>' +
        '<td><span class="tag ' + (consistent ? 'ok' : 'warn') + '">' + (consistent ? '一致' : '需复核') + '</span></td>' +
        '<td><button class="btn small ghost" data-replay="' + i + '">复述重放</button></td></tr>';
    }).join('');
    $('historyBox').innerHTML = '<table><thead><tr><th>时间</th><th>通道</th><th>员工</th><th>场景</th><th>决策码</th><th>结果</th><th>真值校验</th><th>操作</th></tr></thead><tbody>' + rows + '</tbody></table>';
  }

  function replay(i) {
    var h = state.history[i]; if (!h) return;
    if (h.cameraOnly && global.CameraLab) {
      global.CameraLab.replay(h);
      switchTab('monitor');
      toast('已重放摄像头抓拍帧的复述');
      return;
    }
    if (global.CameraLab) global.CameraLab.deactivate();
    if (h.imageOnly && global.ImageLab) {
      global.ImageLab.replay(h);
      switchTab('image');
      toast('已重放图片检测复述');
      return;
    }
    applyConfig(h.cfg); state.last = h;
    renderAll(h); switchTab('monitor');
    toast('已重放 ' + h.result.ts + ' 的检测复述');
  }

  /* ---------------- 对比工作台 ---------------- */
  function evalScene(sid, staff, seed) {
    var base = find(D.SCENES, 'id', sid);
    var rs = E.resolveScene(base, staff);
    var r = E.infer({ scene: rs, staff: staff, seed: seed, channel: state.channel });
    var dec = E.decide(r, { thresholds: state.thresholds, enabled: state.enabled });
    return { base: base, scene: rs, result: r, decision: dec };
  }

  function sideHtml(o) {
    var r = o.result, dec = o.decision;
    var rows = dec.checks.map(function (x) {
      return '<div class="mini-row"><span class="order" style="width:16px;height:16px;font-size:10px">' + x.order + '</span>' +
        '<span>' + x.label + '</span><span class="m" style="color:' + (x.skipped ? '#94a3b8' : x.pass ? '#167a55' : '#c0392b') + '">' +
        (x.skipped ? '跳过' : x.pass ? '通过' : '未通过') + '</span></div>';
    }).join('');
    return '<div class="verdict ' + (dec.allow ? 'allow' : 'deny') + '" style="margin:0 0 10px">' +
      '<div class="v-code">' + dec.code + '</div><div class="v-text">' + (dec.allow ? '放行' : '拦截') + '</div>' +
      '<div class="v-sub">' + o.base.desc + '</div></div>' +
      '<div class="mini-checks">' + rows + '</div>' +
      '<div class="divider"></div>' +
      '<table style="font-size:11.5px"><tbody>' +
      '<tr><td style="color:#64748b">佩戴帽色</td><td class="num">' + D.COLORS[r.color.predicted].name + ' / 授权 ' + D.COLORS[r.color.authorized].name + '</td></tr>' +
      '<tr><td style="color:#64748b">破损概率</td><td class="num">' + fx(r.damage.prob, 3) + '（' + r.damage.cls + '）</td></tr>' +
      '<tr><td style="color:#64748b">佩戴状态</td><td class="num">' + r.wear.label + ' · ' + fx(r.wear.conf) + '</td></tr>' +
      '<tr><td style="color:#64748b">人脸相似度</td><td class="num">' + fx(r.face.score, 3) + ' · 活体' + (r.face.live ? '通过' : '失败') + '</td></tr>' +
      '</tbody></table>';
  }

  function renderCompare() {
    var staff = find(D.STAFF, 'id', $('selCompareStaff').value);
    var A = evalScene($('selSceneA').value, staff, 101);
    var B = evalScene($('selSceneB').value, staff, 202);
    $('tagA').textContent = A.base.label; $('tagB').textContent = B.base.label;
    $('sideA').innerHTML = sideHtml(A); $('sideB').innerHTML = sideHtml(B);
    var diff = 0;
    A.decision.checks.forEach(function (x, i) { if (x.pass !== B.decision.checks[i].pass) diff++; });
    var leaks = [];
    if (A.decision.allow && !A.scene.expectedAllow) leaks.push('A 漏放违规');
    if (B.decision.allow && !B.scene.expectedAllow) leaks.push('B 漏放违规');
    $('diffBox').innerHTML = '<b>对比结论：</b>两侧共 <b>' + diff + ' 项</b>规则判定不同。' +
      'A → <b>' + A.decision.code + '</b>，B → <b>' + B.decision.code + '</b>。' +
      (A.decision.allow !== B.decision.allow
        ? '差异根因：' + (A.decision.failed.length ? A.decision.failed.map(function (x) { return x.label; }).join('、') : '无') +
          ' / ' + (B.decision.failed.length ? B.decision.failed.map(function (x) { return x.label; }).join('、') : '无') + '。'
        : '两项场景最终结论一致，但逐项校验路径不同，可用于验证规则链的可解释性。') +
      (leaks.length ? ' <span style="color:#c0392b">风险：' + leaks.join('、') + '</span>。' : ' 无漏放风险。') +
      ' 复述：' + (A.decision.allow !== B.decision.allow
        ? '系统能在「戴了帽」与「戴得合规」之间做出区分，这正是四任务方案相对单一检测的价值。'
        : '决策引擎按同一顺序串联五项规则，结果可逐条追溯。');
  }

  function paramRange(p) {
    return { damageProb: [0.05, 0.99], faceScore: [0.30, 0.99], wearConf: [0.20, 0.99], colorConf: [0.20, 0.99] }[p] || [0.05, 0.99];
  }

  function renderSweep() {
    var p = $('selSweepParam').value, sid = $('selSweepScene').value;
    var staff = find(D.STAFF, 'id', $('selCompareStaff').value);
    var rg = paramRange(p);
    var sw = E.sweep(p, sid, staff.id, rg[0], rg[1], 21);
    var x0 = 62, x1 = 640, yAllow = 62, yDeny = 126;
    var pts = sw.points.map(function (pt, i) {
      return (x0 + (x1 - x0) * i / (sw.points.length - 1)).toFixed(1) + ',' + (pt.allow ? yAllow : yDeny);
    }).join(' ');
    var bands = '<rect x="' + x0 + '" y="34" width="' + (x1 - x0) + '" height="56" fill="#e6f4ec" opacity=".65"/>' +
      '<rect x="' + x0 + '" y="98" width="' + (x1 - x0) + '" height="56" fill="#fceceb" opacity=".65"/>' +
      '<text x="' + (x0 + 8) + '" y="50" font-size="11" fill="#167a55">放行区 ALLOW</text>' +
      '<text x="' + (x0 + 8) + '" y="114" font-size="11" fill="#c0392b">拦截区 DENY</text>' +
      '<path d="M' + x0 + ' 90 H' + x1 + '" stroke="#cbd5e1" stroke-width="1" stroke-dasharray="4 4"/>';
    var ticks = '';
    for (var t = 0; t <= 4; t++) {
      var tx = x0 + (x1 - x0) * t / 4;
      var tv = (rg[0] + (rg[1] - rg[0]) * t / 4).toFixed(2);
      ticks += '<path d="M' + tx + ' 154 V160" stroke="#94a3b8"/><text x="' + tx + '" y="174" text-anchor="middle" font-size="10.5" fill="#64748b">' + tv + '</text>';
    }
    var dots = sw.points.map(function (pt, i) {
      var xx = x0 + (x1 - x0) * i / (sw.points.length - 1);
      return '<circle cx="' + xx.toFixed(1) + '" cy="' + (pt.allow ? yAllow : yDeny) + '" r="2.6" fill="#2563a8"/>';
    }).join('');
    $('sweepSvg').innerHTML = '<rect x="0" y="0" width="680" height="190" fill="#fff"/>' +
      '<text x="14" y="20" font-size="12" font-weight="600" fill="#1e293b">' + TH_META[p].label + ' · 场景「' + sw.scene + '」判定随阈值变化</text>' +
      bands + '<path d="M' + x0 + ' 90 H' + x1 + '" stroke="#cbd5e1"/>' +
      '<polyline points="' + pts + '" fill="none" stroke="#2563a8" stroke-width="2.2" stroke-linejoin="round"/>' + dots + ticks +
      '<text x="' + ((x0 + x1) / 2) + '" y="188" text-anchor="middle" font-size="10.5" fill="#94a3b8">阈值由严格 → 宽松</text>';

    var flips = [];
    for (var i = 1; i < sw.points.length; i++) if (sw.points[i].allow !== sw.points[i - 1].allow) flips.push(sw.points[i]);
    var critical = flips.length ? flips[flips.length - 1].x : null;
    var r = sw.result;
    $('sweepNote').innerHTML = '<b>灵敏度复述：</b>场景「' + sw.scene + '」的 ' +
      (p === 'damageProb' ? '破损概率 ' + fx(r.damage.prob, 3) : p === 'faceScore' ? '人脸相似度 ' + fx(r.face.score, 3) : '对应置信度') +
      ' 为固定观测值。' +
      (critical !== null
        ? '当阈值放宽到约 <b>' + fx(critical) + '</b> 时判定由拦截翻转为放行，' +
          (sw.points[sw.points.length - 1].allow && !find(D.SCENES, 'id', sid).expectedAllow
            ? '<span style="color:#c0392b">该翻转会造成违规漏放，属于危险配置。</span>'
            : '该翻转属于合理放行。')
        : '在扫描区间内判定未发生翻转，说明当前阈值配置对该场景稳定。') +
      ' 生产中破损阈值应优先保召回，不宜为降低误报而过度放宽。';
  }

  function renderBatch(n, showToast) {
    var b = E.batch(n, 20261123, state.thresholds);
    var A = b.stats.full, B = b.stats.detectOnly;
    var groups = [
      { key: 'allow', label: '放行' }, { key: 'deny', label: '拦截' },
      { key: 'leak', label: '漏放违规' }, { key: 'falseReject', label: '误拒合规' }
    ];
    var max = 1;
    groups.forEach(function (g) { max = Math.max(max, A[g.key], B[g.key]); });
    var baseY = 168, plotH = 118, x0 = 66, gw = 148;
    var grid = '';
    for (var t = 0; t <= 4; t++) {
      var gy = baseY - plotH * t / 4;
      grid += '<path d="M' + x0 + ' ' + gy + ' H660" stroke="#e8edf3" stroke-width="1"/>' +
        '<text x="' + (x0 - 8) + '" y="' + (gy + 4) + '" text-anchor="end" font-size="10" fill="#94a3b8">' + Math.round(max * t / 4) + '</text>';
    }
    var bars = '';
    groups.forEach(function (g, i) {
      var gx = x0 + 18 + i * gw;
      [A, B].forEach(function (S, j) {
        var v = S[g.key], bh = Math.max(2, plotH * v / max);
        var bx = gx + j * 42;
        var color = j === 0 ? (g.key === 'leak' ? '#167a55' : '#2563a8') : (g.key === 'leak' ? '#c0392b' : '#a8b6c8');
        bars += '<rect x="' + bx + '" y="' + (baseY - bh) + '" width="34" height="' + bh + '" rx="3" fill="' + color + '"/>';
        bars += '<text x="' + (bx + 17) + '" y="' + (baseY - bh - 5) + '" text-anchor="middle" font-size="10.5" font-weight="600" fill="#334155">' + v + '</text>';
      });
      bars += '<text x="' + (gx + 42) + '" y="186" text-anchor="middle" font-size="11" fill="#64748b">' + g.label + '</text>';
    });
    var legend = '<rect x="470" y="12" width="12" height="12" rx="2" fill="#2563a8"/><text x="488" y="22" font-size="11" fill="#64748b">完整四任务</text>' +
      '<rect x="574" y="12" width="12" height="12" rx="2" fill="#a8b6c8"/><text x="592" y="22" font-size="11" fill="#64748b">仅检测有无帽</text>';
    $('batchSvg').innerHTML = '<rect x="0" y="0" width="680" height="210" fill="#fff"/>' + legend + grid + bars;

    var rows = [A, B].map(function (S, i) {
      return '<tr><td>' + S.label + '</td><td class="num">' + S.allow + '</td><td class="num">' + S.deny + '</td>' +
        '<td class="num"><span class="tag ' + (S.leak ? 'no' : 'ok') + '">' + S.leak + '</span></td>' +
        '<td class="num">' + S.falseReject + '</td><td class="num">' + fx(100 * S.leak / b.n, 1) + '%</td></tr>';
    }).join('');
    $('batchTable').innerHTML = '<table style="margin-top:12px"><thead><tr><th>策略</th><th>放行</th><th>拦截</th><th>漏放违规</th><th>误拒合规</th><th>漏放率</th></tr></thead><tbody>' + rows + '</tbody></table>';
    $('batchNote').innerHTML = '<b>批量复述：</b>' + b.conclusion;
    if (showToast) toast('已完成 ' + n + ' 次仿真对比');
  }

  /* ---------------- 事件 ---------------- */
  function switchTab(name) {
    state.tab = name;
    Array.prototype.forEach.call(document.querySelectorAll('.tab-btn'), function (b) {
      b.classList.toggle('active', b.dataset.tab === name);
    });
    Array.prototype.forEach.call(document.querySelectorAll('.tab-panel'), function (p) {
      p.classList.toggle('active', p.id === 'tab-' + name);
    });
    /* 隐私：离开实时监控页立即停流，回来时再恢复 */
    if (global.CameraLab) {
      if (name === 'monitor') global.CameraLab.resume();
      else global.CameraLab.pause();
    }
  }

  function toast(msg) {
    var t = $('toast'); t.textContent = msg; t.classList.add('show');
    clearTimeout(toast._t); toast._t = setTimeout(function () { t.classList.remove('show'); }, 2200);
  }

  /* 读取推理服务 /model 的真实权重元信息，填入「系统说明」页卡片 */
  function refreshModelInfo(url, quiet) {
    var box = document.getElementById('modelInfoBox');
    if (!box) return;
    if (!url) {
      box.textContent = '尚未连接推理服务。启动 inference/yolov8_service.py 后，在控制台执行 App.setSource(\'http://127.0.0.1:8000\')。';
      return;
    }
    box.textContent = '正在读取 ' + url + '/model ...';
    fetch(url + '/model').then(function (r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    }).then(function (j) {
      var names = j.runtime_names
        ? Object.keys(j.runtime_names).map(function (k) { return j.runtime_names[k]; })
        : (j.class_names || []);
      box.textContent = [
        '权重文件：' + (j.path || 'anquanmao.pt'),
        '加载状态：' + (j.loaded ? '已加载（真实推理生效）' : '未加载'),
        '任务 / 架构：' + (j.runtime_task || j.task || 'detect') + ' / ' + (j.architecture || 'DetectionModel'),
        '训练环境：Ultralytics ' + (j.version || '-') + ' · ' + (j.date || '-'),
        '运行依赖：torch ' + (j.torch || '-') + ' / ultralytics ' + (j.ultralytics || '-'),
        '类别（' + names.length + '）：' + names.join(' / '),
        '覆盖：' + (j.covers || []).join('、'),
        '不覆盖：' + (j.not_covered || []).join('、'),
        '颜色：' + j.color_method + '；破损：' + j.damage_method,
        '数据集：' + j.dataset
      ].join('\n');
      if (!quiet) toast('已读取真实权重元信息');
    }).catch(function (e) {
      box.textContent = '读取失败：' + e.message + '\n请确认服务已启动、端口正确且 CORS 可用。';
      if (!quiet) toast('模型信息读取失败');
    });
  }

  function bindEvents() {
    Array.prototype.forEach.call(document.querySelectorAll('.tab-btn'), function (b) {
      b.addEventListener('click', function () { switchTab(b.dataset.tab); });
    });

    $('selChannel').addEventListener('change', function () { state.channel = this.value; });
    $('selStaff').addEventListener('change', function () { state.staffId = this.value; });
    $('selScene').addEventListener('change', function () {
      state.sceneId = this.value; state.overrides = null;
      syncControls(find(D.SCENES, 'id', state.sceneId));
      state.seed += 1; runDetection({ silent: false });
    });
    ['selColor', 'selDamage', 'selWear', 'chkLive', 'chkHelmet'].forEach(function (id) {
      $(id).addEventListener('change', function () { readOverrides(); state.seed += 1; runDetection({ silent: true }); });
    });
    $('rngFace').addEventListener('input', function () {
      $('faceScoreVal').textContent = fx(this.value); readOverrides();
    });
    $('rngFace').addEventListener('change', function () { state.seed += 1; runDetection({ silent: true }); });

    $('thresholdBox').addEventListener('input', function (e) {
      var k = e.target.dataset.th; if (!k) return;
      state.thresholds[k] = parseFloat(e.target.value);
      $('val_' + k).textContent = fx(state.thresholds[k]);
      refreshFromLast();
    });
    $('thresholdBox').addEventListener('change', function () { renderSweep(); renderCompare(); });

    $('enabledBox').addEventListener('change', function (e) {
      var k = e.target.dataset.en; if (!k) return;
      state.enabled[k] = e.target.checked;
      refreshFromLast();
      renderCompare();
    });

    $('btnFull').addEventListener('click', function () {
      state.enabled = Object.assign({}, E.STRATEGIES.full.enabled); syncThresholdsOnly(); toast('已切换到完整四任务方案');
    });
    $('btnDetectOnly').addEventListener('click', function () {
      state.enabled = Object.assign({}, E.STRATEGIES.detectOnly.enabled); syncThresholdsOnly(); toast('已切换到「仅检测有无帽」对照策略');
    });
    $('btnRandom').addEventListener('click', function () {
      var s = D.SCENES[Math.floor(Math.random() * D.SCENES.length)];
      state.sceneId = s.id; state.overrides = null; $('selScene').value = s.id;
      syncControls(s); state.seed += 1; runDetection({ silent: false });
    });
    $('btnReset').addEventListener('click', function () {
      if (global.CameraLab) global.CameraLab.deactivate();
      state.thresholds = Object.assign({}, D.DEFAULT_THRESHOLDS);
      state.enabled = Object.assign({}, E.STRATEGIES.full.enabled);
      state.overrides = null; state.source = null;
      syncControls(find(D.SCENES, 'id', state.sceneId));
      state.seed = 1; runDetection({ silent: false }); renderCompare(); renderSweep();
      toast('参数已重置');
    });
    $('btnRun').addEventListener('click', function () { state.seed += 1; runDetection({ silent: false }); toast('检测完成，已写入台账'); });
    $('btnBatch').addEventListener('click', function () { switchTab('compare'); renderBatch(parseInt($('selBatchN').value, 10), true); });

    Array.prototype.forEach.call($('viewSeg').querySelectorAll('button'), function (b) {
      b.addEventListener('click', function () {
        state.view = b.dataset.view;
        Array.prototype.forEach.call($('viewSeg').querySelectorAll('button'), function (x) { x.classList.remove('active'); });
        b.classList.add('active');
        if (state.last) renderScene(state.last.result, state.last.scene, state.last.decision);
      });
    });
    $('chkBoxes').addEventListener('change', function () { state.showBoxes = this.checked; if (state.last) renderScene(state.last.result, state.last.scene, state.last.decision); });
    $('chkKeypoints').addEventListener('change', function () { state.showKeypoints = this.checked; if (state.last) renderScene(state.last.result, state.last.scene, state.last.decision); });

    $('btnCopy').addEventListener('click', function () {
      if (!state.last) return;
      var txt = state.last.narration.full;
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(txt).then(function () { toast('复述已复制'); }, function () { fallbackCopy(txt); });
      } else fallbackCopy(txt);
    });
    $('btnExport').addEventListener('click', function () {
      if (!state.last) return;
      var c = state.last;
      var lines = [
        '工地入口安全帽佩戴检测报告',
        '生成时间：' + new Date().toLocaleString('zh-CN'),
        '帧 ID：frame_' + c.result.seed + ' | 通道：' + c.result.channel,
        '员工：' + c.staff.name + '（' + c.staff.id + '） | 工种：' + c.staff.role,
        '场景：' + c.scene.label,
        '推理源：' + (c.cameraOnly ? '本机摄像头（实时像素分析 + 复核）' : state.source ? '真实 YOLOv8 服务' : '仿真推理'),
        '', c.narration.full, '', '--- 决策明细 ---'
      ];
      if (c.cameraOnly && c.cam) {
        lines.push('数据源：本机摄像头实时帧 #' + c.cam.frame + '；颜色与候选区域由 Canvas HSV 真实像素统计得出，破损 / 佩戴 / 人脸为操作员复核');
      }
      c.decision.checks.forEach(function (x) {
        lines.push(x.order + '. ' + x.label + '：' + (x.skipped ? '跳过' : x.pass ? '通过' : '未通过') + ' | ' + x.detail);
      });
      lines.push('', '--- 参数快照 ---');
      lines.push(JSON.stringify(c.cfg, null, 2));
      var blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
      var a = document.createElement('a');
      a.href = URL.createObjectURL(blob);
      a.download = 'helmet_report_' + c.result.ts.replace(/:/g, '') + '.txt';
      a.click(); URL.revokeObjectURL(a.href);
      toast('报告已导出');
    });

    $('selSceneA').addEventListener('change', renderCompare);
    $('selSceneB').addEventListener('change', renderCompare);
    $('selCompareStaff').addEventListener('change', function () { renderCompare(); renderSweep(); });
    $('selSweepParam').addEventListener('change', renderSweep);
    $('selSweepScene').addEventListener('change', renderSweep);
    $('btnBatchRun').addEventListener('click', function () { renderBatch(parseInt($('selBatchN').value, 10), true); });

    $('historyBox').addEventListener('click', function (e) {
      var idx = e.target.dataset.replay !== undefined ? e.target.dataset.replay : e.target.closest('tr') ? e.target.closest('tr').dataset.idx : null;
      if (idx !== null && idx !== undefined) replay(parseInt(idx, 10));
    });
    $('btnClearHistory').addEventListener('click', function () { state.history = []; renderHistory(); toast('台账已清空'); });

    var btnModel = document.getElementById('btnModelInfo');
    if (btnModel) btnModel.addEventListener('click', function () {
      refreshModelInfo(state.source ? state.source.url : '', false);
      if (!state.source) toast('尚未连接推理服务，请先执行 App.setSource()');
    });
  }

  function fallbackCopy(txt) {
    var ta = document.createElement('textarea'); ta.value = txt; document.body.appendChild(ta);
    ta.select(); try { document.execCommand('copy'); toast('复述已复制'); } catch (e) { toast('复制失败，请使用导出报告'); }
    document.body.removeChild(ta);
  }

  function startClock() {
    function tick() {
      var d = new Date();
      function p(n) { return n < 10 ? '0' + n : '' + n; }
      $('clockChip').textContent = p(d.getHours()) + ':' + p(d.getMinutes()) + ':' + p(d.getSeconds());
    }
    tick(); setInterval(tick, 1000);
  }

  global.App = {
    state: state,
    run: runDetection,
    toast: toast,
    /* 外部数据源（本机摄像头）复用四任务 / 决策 / 复述面板，但不触碰仿真 SVG 画面 */
    applyExternal: function (ctx) {
      state.last = ctx;
      renderPipeline(ctx.result);
      renderDecision(ctx.decision);
      renderNarrative(ctx.narration);
      $('channelChip').textContent = ctx.result.channel;
      $('frameChip').textContent = ctx.cameraOnly && ctx.cam ? 'CAM · 帧 ' + ctx.cam.frame : 'frame_' + ctx.result.seed;
      $('pipelineSub').textContent = ctx.scene.label + ' · ' + ctx.result.ts;
      $('sourceChip').textContent = ctx.cameraOnly ? '推理源：本机摄像头（像素分析）' : (state.source ? '推理源：真实 YOLOv8 服务' : '推理源：仿真推理');
      $('modeChip').className = 'chip ' + (ctx.cameraOnly ? 'green' : state.source ? 'green' : 'amber');
      $('modeChip').innerHTML = '<span class="dot"></span>' +
        (ctx.cameraOnly ? '实时摄像头模式' : state.source ? '真实模型接入' : '演示仿真模式');
    },
    addHistory: function (ctx) {
      state.history.unshift(ctx);
      if (state.history.length > 80) state.history.pop();
      renderHistory();
    },
    setSource: function (url) {
      state.source = new E.RemoteYoloSource(url);
      state.seed += 1;
      runDetection({ silent: false });
      refreshModelInfo(url, true);
      toast('已切换推理源：' + url);
      return state.source;
    },
    modelInfo: refreshModelInfo,
    resetSource: function () {
      state.source = null; state.seed += 1; runDetection({ silent: false });
      refreshModelInfo('', true); toast('已恢复仿真推理源');
    }
  };

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(window);
