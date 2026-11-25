/* 本机摄像头实时演示模块
 *
 * 定位：只替换「实时监控」的画面输入源，仿真模拟（12 场景 / 阈值 / 批量实验）完全不动。
 *
 * 真实计算：getUserMedia 取流 → 每帧 drawImage 到 720×400 → 复用 ImageLab.analyze()
 *           做 HSV 色相直方图 + 12×8 网格连通聚类，输出主色与疑似安全帽区域。
 * 复核项：破损 / 佩戴方式 / 人脸活体无法由色相统计可靠推断，由实时复核表单填写，
 *         模型名与复述文本均明确标注来源，不伪装成 YOLO 输出。
 * 可选全自动：App.setSource() 接入真实 YOLOv8 服务后，每帧以 image_b64 POST /infer。
 * 隐私：帧只在浏览器内存处理，不上传、不录制；离开「实时监控」页签自动停流。
 */
(function (global) {
  'use strict';

  var D = global.SITE_DATA, E = global.HelmetEngine;
  function $(id) { return document.getElementById(id); }
  function find(arr, k, v) { for (var i = 0; i < arr.length; i++) if (arr[i][k] === v) return arr[i]; return arr[0]; }
  function fx(v, n) { return Number(v).toFixed(n === undefined ? 2 : n); }
  var W = 720, H = 400;

  var st = {
    mode: 'simulation', uiOwned: false, active: false, paused: false, busy: false,
    stream: null, video: null, cap: null, capCtx: null, overlay: null, ovCtx: null,
    timer: null, fps: 2, frames: 0, seed: 8100,
    analysis: null, boxes: [], decision: null, last: null,
    scene: null, staff: null, leadText: '',
    showBoxes: true, showGrid: false, autoCapture: true,
    lastDeny: false, lastCaptureAt: 0, yoloFail: false
  };

  function isActive() { return st.uiOwned; }

  /* <svg> 不是 HTMLElement，没有 hidden 这个 IDL 属性，必须走 attribute，否则不会生效 */
  function hide(id, on) {
    var el = $(id); if (!el) return;
    if (on) el.setAttribute('hidden', '');
    else el.removeAttribute('hidden');
  }

  /* ---------------- DOM 切换 ---------------- */
  function setChrome(on) {
    Array.prototype.forEach.call($('dataSourceSeg').querySelectorAll('button'), function (b) {
      b.classList.toggle('active', b.dataset.source === (on ? 'camera' : 'simulation'));
    });
    hide('cameraControls', !on);
    hide('camMetricsCard', !on);
    hide('camLblBoxes', !on);
    hide('camLblGrid', !on);
    hide('lblBoxes', on);
    hide('lblKeypoints', on);
    hide('viewSeg', on);
    hide('cameraStage', !on);
    hide('sceneSvg', on);
    hide('sceneLegend', on);
    hide('camLegend', !on);
    $('sourceHint').textContent = on
      ? '本机摄像头：颜色与候选区域来自真实帧像素；仿真实验数据在「对比工作台」中仍可独立运行。'
      : '仿真模拟：12 个场景预设、阈值、规则开关与批量实验数据全部保留，不受摄像头影响。';
  }

  function resetStage() {
    hide('cameraVideo', false);
    hide('cameraStill', true);
    hide('cameraOverlay', false);
    $('cameraBadge').textContent = '● LIVE · 本机摄像头';
  }

  function setStatus(msg, kind) {
    var el = $('camStatus');
    el.textContent = msg;
    el.className = 'cam-status' + (kind ? ' ' + kind : '');
  }

  function fillSelect(id, items, value) {
    var el = $(id);
    el.innerHTML = items.map(function (it) {
      return '<option value="' + it.value + '">' + it.label + '</option>';
    }).join('');
    if (value !== undefined) el.value = value;
  }

  /* ---------------- 取流 / 停流 ---------------- */
  function start() {
    if (!global.navigator || !global.navigator.mediaDevices || !global.navigator.mediaDevices.getUserMedia) {
      fail('当前环境无法调用摄像头：浏览器未提供 getUserMedia。请通过 https 或 localhost 打开页面（file:// 直开常被禁用），或改用 Edge / Chrome。');
      return;
    }
    setStatus('正在请求摄像头权限…', 'warn');
    global.navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false
    }).then(function (stream) {
      st.stream = stream; st.active = true; st.paused = false; st.yoloFail = false;
      st.video.srcObject = stream;
      return st.video.play().catch(function () { /* 某些无头环境 play() 被拒，但帧仍可读 */ });
    }).then(function () {
      resetStage();
      $('btnCamStart').disabled = true;
      $('btnCamStop').disabled = false;
      setStatus('摄像头已开启 · ' + st.fps + ' fps 实时分析 · 帧仅在浏览器内存中处理，不上传、不录制', 'ok');
      schedule();
      tick();
    }).catch(function (err) {
      fail(camError(err));
    });
  }

  function camError(err) {
    var n = err && err.name ? err.name : '';
    if (n === 'NotAllowedError' || n === 'SecurityError') return '摄像头权限被拒绝。请在浏览器地址栏允许摄像头访问后重试；file:// 直接打开时请改用本地静态服务。';
    if (n === 'NotFoundError' || n === 'DevicesNotFoundError') return '未检测到摄像头设备。请确认设备已连接，或切回「仿真模拟」。';
    if (n === 'NotReadableError') return '摄像头被其他程序占用，请关闭占用程序后重试。';
    return '摄像头开启失败：' + (err && err.message ? err.message : '未知错误') + '。可切回「仿真模拟」。';
  }

  function fail(msg) {
    stopStream();
    st.frames = 0;
    setStatus(msg, 'err');
    $('btnCamStart').disabled = false;
    $('btnCamStop').disabled = true;
    $('cameraBadge').textContent = '● 摄像头不可用';
    $('camMetrics').innerHTML = '<div class="cam-warn">' + msg +
      '<br><br>「仿真模拟」仍然完整可用：12 个场景预设、阈值灵敏度扫描与批量 A/B 实验数据不受影响，点击左上角「仿真模拟」即可返回。</div>';
    global.App.toast('摄像头不可用，已保留仿真模式');
  }

  function stopStream() {
    if (st.timer) { clearInterval(st.timer); st.timer = null; }
    if (st.stream) {
      try { st.stream.getTracks().forEach(function (t) { t.stop(); }); } catch (e) { }
      st.stream = null;
    }
    if (st.video) {
      try { st.video.pause(); } catch (e) { }
      try { st.video.srcObject = null; } catch (e) { }
    }
    st.active = false; st.busy = false;
  }

  function stop() {
    stopStream(); st.paused = false;
    $('btnCamStart').disabled = false;
    $('btnCamStop').disabled = true;
    $('cameraBadge').textContent = '● 已停止';
    setStatus('摄像头已停止 · 点击「开启摄像头」可重新启动', 'warn');
  }

  function pause() {
    if (!st.uiOwned || !st.active) return;
    stopStream(); st.paused = true;
    $('btnCamStart').disabled = false;
    $('cameraBadge').textContent = '⏸ 已暂停（离开监控页）';
    setStatus('已离开「实时监控」页，摄像头自动停流。返回该页会自动恢复。', 'warn');
  }

  function resume() {
    if (st.uiOwned && st.paused && !st.active) start();
  }

  function schedule() {
    if (st.timer) clearInterval(st.timer);
    st.timer = setInterval(function () { tick(); }, Math.round(1000 / st.fps));
  }

  /* ---------------- 模式切换 ---------------- */
  function setMode(name) {
    if (name === 'camera') {
      if (st.uiOwned && st.active) return;
      setChrome(true);
      st.mode = 'camera'; st.uiOwned = true;
      $('cameraBadge').textContent = '● LIVE · 本机摄像头';
      start();
    } else {
      deactivate();
    }
  }

  function deactivate() {
    if (!st.uiOwned) return;
    stopStream();
    st.uiOwned = false; st.active = false; st.paused = false; st.mode = 'simulation';
    setChrome(false);
    $('btnCamStart').disabled = false;
    $('btnCamStop').disabled = true;
    setStatus('未开启 · 视频帧仅在浏览器内存中处理', '');
    /* 立刻恢复顶栏，避免等待异步 renderAll 时出现短暂的状态串台 */
    $('modeChip').className = 'chip amber';
    $('modeChip').innerHTML = '<span class="dot"></span>演示仿真模式';
    $('sourceChip').textContent = '推理源：仿真推理';
    $('frameChip').textContent = 'frame_--';
    global.App.run({ silent: true });
  }

  /* ---------------- 逐帧分析 ---------------- */
  function grab() {
    var v = st.video;
    if (!v || !v.videoWidth || !v.videoHeight) return null;
    var scale = Math.max(W / v.videoWidth, H / v.videoHeight);
    var dw = v.videoWidth * scale, dh = v.videoHeight * scale;
    st.capCtx.clearRect(0, 0, W, H);
    st.capCtx.drawImage(v, (W - dw) / 2, (H - dh) / 2, dw, dh);
    return st.capCtx.getImageData(0, 0, W, H);
  }

  function readForm() {
    return {
      helmet: $('camHelmet').checked, onHead: $('camOnHead').checked,
      color: $('camColor').value, damage: $('camDamage').value, wear: $('camWear').value,
      live: $('camLive').checked, faceScore: parseFloat($('camFace').value)
    };
  }

  function buildBoxes(a, f) {
    if (!f.helmet || !a || !a.region) return [];
    var r = a.region;
    return [{
      cls: 'helmet', label: '疑似安全帽 · ' + D.COLORS[a.main].name,
      box: { x: r.x, y: r.y, w: r.w, h: r.h }, conf: a.conf
    }];
  }

  function damageBoxes(boxes, f) {
    if (f.damage === 'intact' || !boxes.length) return [];
    var b = boxes[0].box;
    return [{
      x: Math.round(b.x + b.w * 0.22), y: Math.round(b.y + b.h * 0.18),
      w: Math.round(b.w * 0.45), h: Math.round(b.h * 0.45), label: '复核：' + D.DAMAGE_LABELS[f.damage]
    }];
  }

  function cameraNarration(nar, dec, scene, staff, frameNo, useYolo, a) {
    /* 直接重写采集段，避免套用「双相机闸机」的仿真措辞 */
    var srcName = useYolo ? '真实 YOLOv8 服务' : '浏览器端 Canvas HSV 像素统计';
    nar.parts[0].text = ('本机摄像头实时取帧 #' + frameNo + '（720 × 400，' + st.fps + ' fps），' +
      '分析源：' + srcName + '，' +
      '主色 ' + D.COLORS[a.main].name + ' H≈' + a.hsvHue + '°（置信度 ' + fx(a.conf, 3) + '）。' +
      '员工 ' + staff.name + '（工号 ' + staff.id + '，' + staff.dept + ' / ' + staff.role + '）进入识别区。视频帧仅在浏览器内存中处理，不上传、不录制。');
    st.leadText = nar.parts[0].text;

    var action = dec.allow ? '闸机开闸，抓拍帧可写入台账。'
      : '闸机保持关闭，声光提示「' + (E.CODE_TEXT[dec.code] || '校验未通过') + '」，抓拍留证并推送安保复核。';
    nar.parts[3].text = ('数据来源：本机摄像头实时帧 #' + frameNo + '（' + srcName +
      ' + 操作员复核）。系统判定：' + (dec.allow ? '允许通行' : '禁止通行') + '。' +
      '复核表单标注的期望结果：' + (scene.expectedAllow ? '允许通行' : '禁止通行') + '；' +
      (dec.allow !== scene.expectedAllow ? '⚠ 判定与复核标注不一致，请检查阈值或复核项。' : '判定与复核标注一致。') + action);
    nar.full = nar.parts.map(function (p) { return '【' + p.title + '】' + p.text; }).join('\n');
  }

  function finish(result, a, f, scene, staff, frameUrl, boxes, dBoxes, useYolo) {
    if (!useYolo) {
      result.source = 'camera';
      /* 像素聚类没有找到候选区时，必须显式清空检测结果：
         buildResult 只在 boxes 非空时覆盖 detection，否则会残留仿真的高置信度框，
         造成「画面里明明没框，四任务面板却显示检出安全帽」的假象。 */
      if (!boxes.length) {
        result.detection.items = [];
        result.detection.boxes = {};
        result.detection.count = 0;
        result.detection.helmetConf = 0;
        result.detection.helmetOnHead = false;
      }
      result.detection.model = '浏览器端候选区（HSV 网格聚类）';
      result.color.model = 'Canvas HSV 像素统计（摄像头实时帧）';
      result.damage.model = '操作员复核（实时摄像头）';
      result.wear.model = '关键点规则 + 操作员复核（实时摄像头）';
      result.face.model = '既有刷脸系统 / 操作员录入';
    }
    result.origin = 'camera';
    result.cameraFrame = st.frames;

    var dec = E.decide(result, {
      thresholds: global.App.state.thresholds, enabled: global.App.state.enabled
    });
    var nar = E.narrate({ result: result, decision: dec, scene: scene, staff: staff });
    cameraNarration(nar, dec, scene, staff, st.frames, useYolo, a);

    var ctx = {
      result: result, decision: dec, narration: nar, scene: scene, staff: staff,
      cameraOnly: true,
      cam: { frame: st.frames, dataUrl: frameUrl, analysis: a, form: f },
      cfg: {
        channel: global.App.state.channel, staffId: staff.id, sceneId: 'S01', overrides: null,
        thresholds: Object.assign({}, global.App.state.thresholds),
        enabled: Object.assign({}, global.App.state.enabled), seed: st.seed
      }
    };
    var overlayBoxes = useYolo ? result.detection.items : boxes;
    var overlayDamage = useYolo ? result.damage.boxes : dBoxes;
    st.last = ctx; st.decision = dec; st.analysis = a; st.boxes = overlayBoxes;
    global.App.applyExternal(ctx);
    drawOverlay(a, overlayBoxes, overlayDamage, dec);
    renderMetrics(a, ctx, useYolo);
    autoCapture(dec);
    st.busy = false;
  }

  function autoCapture(dec) {
    var now = Date.now(), deny = !dec.allow;
    if (deny && !st.lastDeny && st.autoCapture && now - st.lastCaptureAt > 4000) {
      st.lastCaptureAt = now;
      global.App.addHistory(Object.assign({}, st.last));
      global.App.toast('检测到拦截帧，已自动留证');
    }
    st.lastDeny = deny;
  }

  function tick() {
    if (!st.uiOwned) return;
    if (!st.active || !st.stream || st.busy || !st.video || st.video.readyState < 2) {
      if (st.last) refresh();
      return;
    }
    st.busy = true;
    var img;
    try { img = grab(); } catch (e) { st.busy = false; return; }
    if (!img) { st.busy = false; return; }

    st.frames += 1; st.seed += 1;
    var a = global.ImageLab.analyze(img);
    var f = readForm();
    var staff = find(D.STAFF, 'id', global.App.state.staffId);
    var color = f.color === 'auto' ? a.main : f.color;
    var scene = E.resolveScene({
      id: 'CAM', label: '本机摄像头实时帧 #' + st.frames,
      desc: '颜色由 Canvas HSV 真实像素统计得出；破损 / 佩戴 / 人脸由实时复核表单标注，真值用于演示一致性校验，非现场实测结论',
      helmetPresent: f.helmet, color: color, damage: f.damage, wear: f.wear,
      live: f.live, faceScore: f.faceScore, expectedAllow: false
    }, staff);
    st.scene = scene; st.staff = staff;

    var boxes = buildBoxes(a, f);
    var dBoxes = damageBoxes(boxes, f);
    var frameUrl = st.cap.toDataURL('image/jpeg', 0.72);
    var view = { scale: 1, ox: 0, oy: 0 };
    var cfgBase = {
      scene: scene, staff: staff, seed: st.seed, channel: global.App.state.channel,
      analysis: {
        hsvHue: a.hsvHue, clsProbs: a.clsProbs, conf: a.conf,
        model: 'Canvas HSV 像素统计（摄像头实时帧）'
      },
      boxes: boxes, damageBoxes: dBoxes, helmetOnHead: f.onHead,
      image: { name: '摄像头实时帧 #' + st.frames, dataUrl: frameUrl },
      view: view, autoColor: f.color === 'auto', origin: 'camera'
    };
    var src = global.App.state.source;

    if (src && !st.yoloFail) {
      E.infer({
        scene: scene, staff: staff, seed: st.seed, channel: global.App.state.channel,
        image_b64: frameUrl, view: view
      }, src).then(function (r) {
        finish(r, a, f, scene, staff, frameUrl, boxes, dBoxes, true);
      }).catch(function () {
        st.yoloFail = true;
        global.App.toast('模型服务不可用，已降级为摄像头像素分析');
        finish(E.buildResult(cfgBase), a, f, scene, staff, frameUrl, boxes, dBoxes, false);
      });
    } else {
      finish(E.buildResult(cfgBase), a, f, scene, staff, frameUrl, boxes, dBoxes, false);
    }
  }

  /* 阈值 / 规则 / 员工等参数变化后，用当前帧重新决策，不重新取流 */
  function refresh() {
    if (!st.last) return;
    var c = st.last;
    c.decision = E.decide(c.result, {
      thresholds: global.App.state.thresholds, enabled: global.App.state.enabled
    });
    c.narration = E.narrate({
      result: c.result, decision: c.decision, scene: c.scene, staff: c.staff
    });
    if (st.leadText) {
      c.narration.parts[0].text = st.leadText;
      c.narration.full = c.narration.parts.map(function (p) { return '【' + p.title + '】' + p.text; }).join('\n');
    }
    st.decision = c.decision;
    global.App.applyExternal(c);
    if (st.analysis) drawOverlay(st.analysis, st.boxes, [], c.decision);
  }

  /* ---------------- 画面叠加 ---------------- */
  function drawOverlay(a, boxes, dBoxes, dec) {
    if (!st.ovCtx) return;
    var c = st.ovCtx;
    c.clearRect(0, 0, W, H);
    c.font = '12px "Microsoft YaHei", sans-serif';
    c.textBaseline = 'middle';

    if (st.showGrid) {
      c.strokeStyle = 'rgba(255,255,255,.22)'; c.lineWidth = 1;
      for (var gx = 1; gx < 12; gx++) { c.beginPath(); c.moveTo(gx * 60, 0); c.lineTo(gx * 60, H); c.stroke(); }
      for (var gy = 1; gy < 8; gy++) { c.beginPath(); c.moveTo(0, gy * 50); c.lineTo(W, gy * 50); c.stroke(); }
    }
    if (st.showBoxes) {
      boxes.forEach(function (it) {
        var b = it.box, label = it.label + ' ' + fx(it.conf);
        c.setLineDash([7, 5]); c.strokeStyle = '#5b4bb7'; c.lineWidth = 2;
        c.strokeRect(b.x, b.y, b.w, b.h); c.setLineDash([]);
        c.fillStyle = '#5b4bb7';
        var tw = c.measureText(label).width + 12;
        c.fillRect(b.x, Math.max(2, b.y - 20), tw, 19);
        c.fillStyle = '#fff'; c.fillText(label, b.x + 6, Math.max(2, b.y - 20) + 10);
      });
      (dBoxes || []).forEach(function (b) {
        c.strokeStyle = '#c0392b'; c.lineWidth = 2; c.strokeRect(b.x, b.y, b.w, b.h);
        c.fillStyle = '#c0392b';
        var tw = c.measureText(b.label).width + 12;
        c.fillRect(b.x, Math.max(2, b.y - 19), tw, 18);
        c.fillStyle = '#fff'; c.fillText(b.label, b.x + 6, Math.max(2, b.y - 19) + 9);
      });
    }

    /* 右上实时指标 */
    c.fillStyle = 'rgba(15,23,42,.72)';
    c.fillRect(W - 208, 10, 196, 50);
    c.fillStyle = '#fff';
    c.fillText('帧 #' + st.frames + ' · 主色 ' + D.COLORS[a.main].name, W - 198, 26);
    c.fillText('H≈' + a.hsvHue + '°  置信 ' + fx(a.conf, 3), W - 198, 45);

    /* 底部结论条 */
    c.fillStyle = dec.allow ? 'rgba(230,244,236,.95)' : 'rgba(252,236,235,.95)';
    c.fillRect(12, 350, 696, 38);
    c.strokeStyle = dec.allow ? '#bfe2ce' : '#f0c4c0'; c.lineWidth = 1;
    c.strokeRect(12, 350, 696, 38);
    c.fillStyle = dec.allow ? '#167a55' : '#c0392b';
    c.beginPath(); c.arc(30, 369, 6, 0, Math.PI * 2); c.fill();
    c.font = 'bold 13px "Microsoft YaHei", sans-serif';
    c.fillText(dec.allow ? 'ALLOW · 闸机放行' : 'DENY · ' + dec.code + ' 拦截', 44, 370);
    c.font = '11px "Microsoft YaHei", sans-serif';
    c.fillStyle = '#475569';
    c.textAlign = 'right';
    c.fillText(global.App.state.channel + ' · ' + st.fps + ' fps · 帧 #' + st.frames, 696, 370);
    c.textAlign = 'left';
  }

  /* ---------------- 实时像素分析面板 ---------------- */
  function renderMetrics(a, ctx, useYolo) {
    var bars = Object.keys(D.COLORS).map(function (k) {
      var p = a.clsProbs[k] || 0;
      return '<div class="cam-bar-row"><span>' + D.COLORS[k].name + '</span>' +
        '<div class="bar"><span style="width:' + Math.round(p * 100) + '%;background:' + D.COLORS[k].hex + '"></span></div>' +
        '<span style="text-align:right">' + fx(p * 100, 1) + '%</span></div>';
    }).join('');
    $('camMetrics').innerHTML =
      '<div class="cam-kv">' +
      '<dt>采样像素</dt><dd>' + a.sampled.toLocaleString() + ' 点 / 帧（命中高饱和 ' + a.matched.toLocaleString() + ' 点）</dd>' +
      '<dt>主色判定</dt><dd><span class="chip" style="background:' + D.COLORS[a.main].hex + ';color:' + D.COLORS[a.main].text + ';border:1px solid ' + D.COLORS[a.main].stroke + '">' +
        D.COLORS[a.main].name + ' · ' + D.COLORS[a.main].grade + '</span> 置信度 ' + fx(a.conf, 3) + '</dd>' +
      '<dt>HSV 主色相</dt><dd>H ≈ ' + a.hsvHue + '°</dd>' +
      '<dt>候选区域</dt><dd>' + (a.region ? a.region.w + ' × ' + a.region.h + ' px，' + a.region.cells + ' 个网格连通' : '未定位到明显盔体') + '</dd>' +
      '<dt>当前判定</dt><dd><span class="tag ' + (ctx.decision.allow ? 'ok' : 'no') + '">' + ctx.decision.code + '</span></dd>' +
      '</div>' +
      '<div class="section-title">色相分布（当前帧真实像素）</div><div class="cam-bars">' + bars + '</div>' +
      (a.lowConfidence ? '<div class="cam-warn">主色置信度偏低：画面可能不含明显安全帽，或存在大面积高饱和背景（安全网、工服）。请在左侧复核表单手动指定帽色。</div>' : '') +
      '<div class="cam-foot">真算：颜色、HSV 主色相、四色分布、疑似安全帽区域定位。需复核：破损 / 下颚带 / 反戴 / 人脸活体。' +
      (useYolo ? '当前帧由真实 YOLOv8 服务推理，HSV 统计仅作辅助展示。' : (global.App.state.source && st.yoloFail ? '真实模型服务曾失败，已降级为像素分析。' : '接入 App.setSource() 后本页将自动切换为模型推理。')) +
      '</div>';
    $('camMetricsSub').textContent = '帧 #' + st.frames + ' · ' + (useYolo ? '真实 YOLOv8 服务' : '浏览器端像素分析 + 复核');
  }

  /* ---------------- 抓拍 / 台账重放 ---------------- */
  function capture() {
    if (!st.last) { global.App.toast('请先开启摄像头并等待至少一帧'); return; }
    global.App.addHistory(Object.assign({}, st.last));
    global.App.toast('已抓拍帧 #' + st.last.cam.frame + ' 并写入通行台账');
  }

  function replay(ctx) {
    stopStream();
    setChrome(true);
    st.mode = 'camera'; st.uiOwned = true; st.active = false; st.paused = false; st.busy = false;
    st.last = ctx; st.decision = ctx.decision;
    st.analysis = ctx.cam && ctx.cam.analysis ? ctx.cam.analysis : null;
    st.boxes = ctx.result.detection ? ctx.result.detection.items : [];
    hide('cameraVideo', true);
    hide('cameraOverlay', true);
    var im = $('cameraStill');
    im.src = ctx.cam && ctx.cam.dataUrl ? ctx.cam.dataUrl : '';
    hide('cameraStill', false);
    $('cameraBadge').textContent = '台账重放 · 抓拍帧 #' + (ctx.cam ? ctx.cam.frame : '-');
    $('btnCamStart').disabled = false;
    $('btnCamStop').disabled = true;
    setStatus('正在重放台账中的抓拍帧，摄像头未开启。点击「开启摄像头」可恢复实时演示。', 'warn');
    global.App.applyExternal(ctx);
    if (st.analysis) renderMetrics(st.analysis, ctx, !!ctx.result.yolo);
  }

  /* ---------------- 初始化 ---------------- */
  function init() {
    st.video = $('cameraVideo');
    st.overlay = $('cameraOverlay');
    st.cap = document.createElement('canvas'); st.cap.width = W; st.cap.height = H;
    try {
      st.capCtx = st.cap.getContext('2d', { willReadFrequently: true });
      st.ovCtx = st.overlay.getContext('2d');
    } catch (e) { st.capCtx = null; st.ovCtx = null; }

    fillSelect('camDamage', Object.keys(D.DAMAGE_LABELS).map(function (k) {
      return { value: k, label: D.DAMAGE_LABELS[k] };
    }), 'intact');
    fillSelect('camWear', Object.keys(D.WEAR_LABELS).map(function (k) {
      return { value: k, label: D.WEAR_LABELS[k] };
    }), 'correct');
    fillSelect('camColor', [{ value: 'auto', label: '自动采用像素分析主色' }].concat(
      Object.keys(D.COLORS).map(function (k) { return { value: k, label: D.COLORS[k].name + ' · ' + D.COLORS[k].grade }; })
    ), 'auto');

    Array.prototype.forEach.call($('dataSourceSeg').querySelectorAll('button'), function (b) {
      b.addEventListener('click', function () { setMode(b.dataset.source); });
    });
    $('btnCamStart').addEventListener('click', function () {
      if (!st.uiOwned) { setMode('camera'); return; }
      start();
    });
    $('btnCamStop').addEventListener('click', stop);
    $('btnCamSnapshot').addEventListener('click', capture);
    $('camFps').addEventListener('change', function () {
      st.fps = parseInt(this.value, 10) || 2;
      if (st.active) schedule();
      setStatus('摄像头已开启 · ' + st.fps + ' fps 实时分析 · 帧仅在浏览器内存中处理，不上传、不录制', 'ok');
    });
    $('camFace').addEventListener('input', function () {
      $('camFaceVal').textContent = fx(this.value);
      $('camFaceLbl').textContent = fx(this.value);
    });
    $('camShowBoxes').addEventListener('change', function () {
      st.showBoxes = this.checked;
      if (st.analysis && st.last) drawOverlay(st.analysis, st.boxes, [], st.last.decision);
    });
    $('camShowGrid').addEventListener('change', function () {
      st.showGrid = this.checked;
      if (st.analysis && st.last) drawOverlay(st.analysis, st.boxes, [], st.last.decision);
    });
    $('camAutoCapture').addEventListener('change', function () { st.autoCapture = this.checked; });
  }

  global.CameraLab = {
    init: init, isActive: isActive, setMode: setMode, deactivate: deactivate,
    start: start, stop: stop, pause: pause, resume: resume,
    tick: tick, refresh: refresh, capture: capture, replay: replay,
    state: st
  };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(window);
