/* 图片检测模块
 * 真实能力：读取导入图片的像素，做 HSV 色相统计与主色判定，并用网格聚类
 *           找出疑似安全帽区域（浏览器端 Canvas 计算，非伪造结果）。
 * 边界：破损 / 佩戴 / 人脸无法由像素统计可靠得出，默认交给操作员复核；
 *       若已通过 App.setSource() 接入真实 YOLOv8 服务，则改由模型自动判定。
 */
(function (global) {
  'use strict';
  var D = global.SITE_DATA, E = global.HelmetEngine;
  function $(id) { return document.getElementById(id); }
  function find(arr, k, v) { for (var i = 0; i < arr.length; i++) if (arr[i][k] === v) return arr[i]; return arr[0]; }
  function fx(v, n) { return Number(v).toFixed(n === undefined ? 2 : n); }
  var W = 720, H = 400, GW = 12, GH = 8, CW = W / GW, CH = H / GH;

  var st = {
    image: null, view: { scale: 1, ox: 0, oy: 0 }, analysis: null,
    boxes: [], damageBoxes: [], showBoxes: true, showGrid: false,
    result: null, decision: null, narration: null, scene: null, staff: null,
    last: null, seed: 9001
  };

  /* ---------------- 像素分析 ---------------- */
  function rgb2hsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    var mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    var h = 0, s = mx === 0 ? 0 : d / mx, v = mx;
    if (d > 0) {
      if (mx === r) h = 60 * (((g - b) / d) % 6);
      else if (mx === g) h = 60 * ((b - r) / d + 2);
      else h = 60 * ((r - g) / d + 4);
    }
    if (h < 0) h += 360;
    return { h: h, s: s, v: v };
  }

  function classify(h, s, v) {
    if (s < 0.16 && v > 0.62) return 'white';
    if (s < 0.42 || v < 0.22) return null;
    if (h < 16 || h >= 340) return 'red';
    if (h >= 34 && h < 72) return 'yellow';
    if (h >= 185 && h < 265) return 'blue';
    return null; /* 16-34 为常见肤色区，72-185 为绿色背景，均不参与帽色统计 */
  }

  function analyze(imgData) {
    var px = imgData.data, scores = { white: 0, yellow: 0, blue: 0, red: 0 };
    var hueAcc = { white: 0, yellow: 0, blue: 0, red: 0 }, wAcc = { white: 0, yellow: 0, blue: 0, red: 0 };
    var grid = [], i, g;
    for (i = 0; i < GW * GH; i++) grid.push({ white: 0, yellow: 0, blue: 0, red: 0 });
    var sampled = 0, matched = 0;
    for (var y = 0; y < H; y += 2) {
      for (var x = 0; x < W; x += 2) {
        var p = (y * W + x) * 4;
        var c = rgb2hsv(px[p], px[p + 1], px[p + 2]);
        sampled++;
        var bucket = classify(c.h, c.s, c.v);
        if (!bucket) continue;
        /* 安全帽通常位于画面上半部；高饱和像素权重更高 */
        var w = c.s * c.s * (y < H * 0.55 ? 1.7 : 0.40);
        scores[bucket] += w; matched++;
        hueAcc[bucket] += c.h * w; wAcc[bucket] += w;
        g = Math.floor(y / CH) * GW + Math.floor(x / CW);
        if (g >= 0 && g < grid.length) grid[g][bucket] += w;
      }
    }
    var total = scores.white + scores.yellow + scores.blue + scores.red;
    var clsProbs = {}, main = 'yellow', max = -1;
    Object.keys(scores).forEach(function (k) {
      clsProbs[k] = total > 0 ? Math.round((scores[k] / total + 0.015) * 1000) / 1000 : 0.25;
      if (scores[k] > max) { max = scores[k]; main = k; }
    });
    var sum = Object.keys(clsProbs).reduce(function (a, k) { return a + clsProbs[k]; }, 0);
    Object.keys(clsProbs).forEach(function (k) { clsProbs[k] = Math.round(clsProbs[k] / sum * 1000) / 1000; });
    var conf = total > 0 ? Math.round(scores[main] / total * 1000) / 1000 : 0;
    var hsvHue = wAcc[main] > 0 ? Math.round(hueAcc[main] / wAcc[main]) : 0;

    /* 网格连通聚类，输出疑似安全帽区域 */
    var thr = Math.max(max * 0.16, 0.001);
    var best = -1, bestVal = 0;
    grid.forEach(function (cell, idx) { if (cell[main] > bestVal) { bestVal = cell[main]; best = idx; } });
    var region = null;
    if (best >= 0 && bestVal > 0) {
      var seen = {}, queue = [best], minX = GW, minY = GH, maxX = -1, maxY = -1;
      while (queue.length) {
        var cur = queue.shift(); if (seen[cur]) continue; seen[cur] = 1;
        if (grid[cur][main] < thr) continue;
        var gx = cur % GW, gy = Math.floor(cur / GW);
        minX = Math.min(minX, gx); maxX = Math.max(maxX, gx);
        minY = Math.min(minY, gy); maxY = Math.max(maxY, gy);
        [[1, 0], [-1, 0], [0, 1], [0, -1]].forEach(function (d) {
          var nx = gx + d[0], ny = gy + d[1];
          if (nx >= 0 && nx < GW && ny >= 0 && ny < GH) queue.push(ny * GW + nx);
        });
      }
      if (maxX >= minX) {
        region = {
          x: Math.round(minX * CW), y: Math.round(minY * CH),
          w: Math.round((maxX - minX + 1) * CW), h: Math.round((maxY - minY + 1) * CH),
          cells: Object.keys(seen).length
        };
      }
    }
    return {
      sampled: sampled, matched: matched, main: main, conf: conf, clsProbs: clsProbs,
      hsvHue: hsvHue, region: region, scores: scores, grid: grid,
      lowConfidence: conf < 0.32,
      model: '浏览器端 Canvas HSV 像素统计'
    };
  }

  /* ---------------- 图片载入 ---------------- */
  function setImage(name, dataUrl, rawUrl, natW, natH, view) {
    st.image = { name: name, dataUrl: dataUrl, rawDataUrl: rawUrl, naturalW: natW, naturalH: natH, view: view };
    st.view = view;
    $('imgMeta').innerHTML = '<b>' + name + '</b><br>原始尺寸 ' + natW + ' × ' + natH +
      ' · 标准化画布 720 × 400 · 缩放 ' + fx(view.scale, 3) + '<br>' +
      '推理源：' + (global.App && global.App.state.source ? '真实 YOLOv8 服务' : '浏览器端像素分析 + 人工复核');
    st.seed += 1;
    buildBoxes();
    renderImage();
    renderAnalysis();
  }

  function normalizeImage(im, name, rawUrl) {
    var natW = im.naturalWidth || 720, natH = im.naturalHeight || 400;
    var cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    var ctx = cv.getContext('2d');
    ctx.fillStyle = '#eef2f7'; ctx.fillRect(0, 0, W, H);
    var scale = Math.max(W / natW, H / natH);
    var dw = natW * scale, dh = natH * scale;
    var ox = (W - dw) / 2, oy = (H - dh) / 2;
    ctx.drawImage(im, ox, oy, dw, dh);
    var dataUrl = cv.toDataURL('image/png');
    st.analysis = analyze(ctx.getImageData(0, 0, W, H));
    setImage(name, dataUrl, rawUrl || dataUrl, natW, natH, { scale: scale, ox: ox, oy: oy });
  }

  function loadFile(file) {
    if (!file || !/^image\//.test(file.type)) { global.App.toast('请选择图片文件（jpg / png / webp）'); return; }
    var fr = new FileReader();
    fr.onload = function () {
      var im = new Image();
      im.onload = function () { normalizeImage(im, file.name, fr.result); global.App.toast('图片已导入，像素分析完成'); };
      im.onerror = function () { global.App.toast('图片解码失败'); };
      im.src = fr.result;
    };
    fr.onerror = function () { global.App.toast('文件读取失败'); };
    fr.readAsDataURL(file);
  }

  function makeSample() {
    var cv = document.createElement('canvas'); cv.width = W; cv.height = H;
    var ctx = cv.getContext('2d');
    var bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, '#dfe6ee'); bg.addColorStop(1, '#aab6c4');
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    ctx.fillStyle = '#8f9dac'; ctx.fillRect(0, 288, W, 112);
    ctx.strokeStyle = '#98a6b5'; ctx.lineWidth = 2;
    for (var i = 0; i < 6; i++) { ctx.beginPath(); ctx.moveTo(i * 140, 400); ctx.lineTo(300 + i * 22, 288); ctx.stroke(); }
    ctx.fillStyle = '#7d8b9c'; ctx.fillRect(40, 150, 40, 150); ctx.fillRect(640, 150, 40, 150);
    /* 人物 */
    ctx.fillStyle = '#5c7089'; ctx.beginPath();
    ctx.moveTo(272, 400); ctx.lineTo(288, 238); ctx.quadraticCurveTo(360, 214, 432, 238); ctx.lineTo(448, 400); ctx.fill();
    ctx.fillStyle = '#f5c518'; ctx.fillRect(288, 268, 144, 12);
    ctx.fillStyle = '#e8b78f'; ctx.fillRect(346, 208, 28, 26);
    ctx.fillStyle = '#f2c9a5'; ctx.beginPath(); ctx.ellipse(360, 178, 34, 42, 0, 0, Math.PI * 2); ctx.fill();
    /* 黄色安全帽（示例图真实像素来源） */
    ctx.fillStyle = '#f5c518'; ctx.beginPath();
    ctx.moveTo(306, 168); ctx.arc(360, 168, 54, Math.PI, 0); ctx.lineTo(414, 172); ctx.lineTo(306, 172); ctx.fill();
    ctx.fillStyle = '#f7cf3c'; ctx.beginPath(); ctx.ellipse(360, 172, 62, 9, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#c99a00'; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(360, 118); ctx.lineTo(360, 168); ctx.stroke();
    /* 下颚带 */
    ctx.strokeStyle = '#9a6a12'; ctx.lineWidth = 2.5; ctx.beginPath();
    ctx.moveTo(306, 172); ctx.quadraticCurveTo(320, 216, 360, 226); ctx.quadraticCurveTo(400, 216, 414, 172); ctx.stroke();
    /* 轻微噪点，让像素统计更接近真实照片 */
    for (var n = 0; n < 900; n++) {
      ctx.fillStyle = 'rgba(255,255,255,' + (Math.random() * 0.06) + ')';
      ctx.fillRect(Math.random() * W, Math.random() * H, 2, 2);
    }
    var dataUrl = cv.toDataURL('image/png');
    st.analysis = analyze(ctx.getImageData(0, 0, W, H));
    setImage('示例图 · 黄色安全帽（像素生成）', dataUrl, dataUrl, W, H, { scale: 1, ox: 0, oy: 0 });
    global.App.toast('示例图片已生成，可立即执行检测');
  }

  /* ---------------- 框与分析面板 ---------------- */
  function buildBoxes() {
    st.boxes = [];
    var a = st.analysis;
    if (!a || !a.region) return;
    var r = a.region;
    st.boxes.push({
      cls: 'helmet', label: '疑似安全帽 · ' + D.COLORS[a.main].name,
      box: { x: r.x, y: r.y, w: r.w, h: r.h }, conf: a.conf
    });
  }

  function candidateDamageBox() {
    if (!st.boxes.length) return [];
    var b = st.boxes[0].box;
    return [{
      x: Math.round(b.x + b.w * 0.22), y: Math.round(b.y + b.h * 0.18),
      w: Math.round(b.w * 0.45), h: Math.round(b.h * 0.45), label: '复核标记'
    }];
  }

  function renderImage() {
    var svg = '';
    if (!st.image) {
      svg = '<rect x="0" y="0" width="720" height="400" fill="#f8fafc"/>' +
        '<rect x="60" y="70" width="600" height="260" rx="14" fill="#fff" stroke="#dbe3ed" stroke-dasharray="6 5"/>' +
        '<circle cx="360" cy="158" r="30" fill="#e8f1fb" stroke="#2563a8" stroke-width="1.5"/>' +
        '<path d="M348 158 H372 M360 146 V170" stroke="#2563a8" stroke-width="2.5" stroke-linecap="round"/>' +
        '<text x="360" y="224" text-anchor="middle" font-size="15" font-weight="600" fill="#1e293b">导入一张工地照片开始检测</text>' +
        '<text x="360" y="250" text-anchor="middle" font-size="12" fill="#64748b">支持点击选择 / 拖拽上传，或点击「生成示例图片」</text>' +
        '<text x="360" y="274" text-anchor="middle" font-size="11.5" fill="#94a3b8">颜色由真实像素统计得出；破损 / 佩戴 / 人脸需复核或接入真实模型</text>';
      $('imageSvg').innerHTML = svg;
      return;
    }
    svg += '<image href="' + st.image.dataUrl + '" x="0" y="0" width="720" height="400" preserveAspectRatio="none"/>';
    if (st.showGrid && st.analysis) {
      for (var gx = 1; gx < GW; gx++) svg += '<path d="M' + (gx * CW) + ' 0 V400" stroke="rgba(255,255,255,.22)" stroke-width="1"/>';
      for (var gy = 1; gy < GH; gy++) svg += '<path d="M0 ' + (gy * CH) + ' H720" stroke="rgba(255,255,255,.22)" stroke-width="1"/>';
    }
    if (st.showBoxes) {
      st.boxes.forEach(function (it) {
        var b = it.box, label = it.label + ' ' + fx(it.conf);
        svg += '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" fill="none" stroke="#5b4bb7" stroke-width="2" stroke-dasharray="6 4"/>';
        svg += '<rect x="' + b.x + '" y="' + (b.y - 18) + '" width="' + (label.length * 8.6 + 12) + '" height="18" rx="4" fill="#5b4bb7"/>';
        svg += '<text x="' + (b.x + 6) + '" y="' + (b.y - 5) + '" font-size="11" fill="#fff">' + label + '</text>';
      });
      st.damageBoxes.forEach(function (b) {
        svg += '<rect x="' + b.x + '" y="' + b.y + '" width="' + b.w + '" height="' + b.h + '" fill="none" stroke="#c0392b" stroke-width="2"/>';
        svg += '<rect x="' + b.x + '" y="' + (b.y - 17) + '" width="' + (b.label.length * 12 + 10) + '" height="17" rx="4" fill="#c0392b"/>';
        svg += '<text x="' + (b.x + 5) + '" y="' + (b.y - 4) + '" font-size="10.5" fill="#fff">' + b.label + '</text>';
      });
    }
    var dec = st.decision;
    var allow = dec ? dec.allow : null;
    svg += '<rect x="12" y="350" width="696" height="38" rx="8" fill="' +
      (allow === null ? '#f1f5f9' : allow ? '#e6f4ec' : '#fceceb') + '" stroke="' +
      (allow === null ? '#cbd5e1' : allow ? '#bfe2ce' : '#f0c4c0') + '"/>' +
      '<text x="28" y="374" font-size="13" font-weight="600" fill="' +
      (allow === null ? '#64748b' : allow ? '#167a55' : '#c0392b') + '">' +
      (allow === null ? '等待检测' : allow ? 'ALLOW · 闸机放行' : 'DENY · ' + dec.code + ' 拦截') + '</text>' +
      '<text x="696" y="374" text-anchor="end" font-size="10.5" fill="#475569">' + st.image.name + '</text>';
    $('imageSvg').innerHTML = svg;
  }

  function renderAnalysis() {
    var a = st.analysis;
    if (!a) {
      $('analysisBox').innerHTML = '<p class="cr-detail" style="margin:0">尚未导入图片。导入后系统会统计画面像素的 HSV 色相分布，定位疑似安全帽区域。</p>';
      return;
    }
    var bars = Object.keys(D.COLORS).map(function (k) {
      var p = a.clsProbs[k] || 0;
      return '<div style="display:grid;grid-template-columns:34px 1fr 44px;gap:8px;align-items:center;margin-bottom:6px">' +
        '<span style="font-size:11.5px">' + D.COLORS[k].name + '</span>' +
        '<div class="bar"><span style="width:' + Math.round(p * 100) + '%;background:' + D.COLORS[k].hex + '"></span></div>' +
        '<span style="font-size:11px;text-align:right;color:#64748b">' + fx(p * 100, 1) + '%</span></div>';
    }).join('');
    $('analysisBox').innerHTML =
      '<div class="kv" style="margin-bottom:10px">' +
      '<dt>采样像素</dt><dd>' + a.sampled.toLocaleString() + ' 点（命中高饱和 ' + a.matched.toLocaleString() + ' 点）</dd>' +
      '<dt>主色判定</dt><dd><span class="chip" style="background:' + D.COLORS[a.main].hex + ';color:' + D.COLORS[a.main].text + ';border:1px solid ' + D.COLORS[a.main].stroke + '">' + D.COLORS[a.main].name + ' · ' + D.COLORS[a.main].grade + '</span></dd>' +
      '<dt>HSV 主色相</dt><dd>H ≈ ' + a.hsvHue + '° · 置信度 ' + fx(a.conf) + '</dd>' +
      '<dt>候选区域</dt><dd>' + (a.region ? a.region.w + ' × ' + a.region.h + ' px，' + a.region.cells + ' 个网格连通' : '未定位到') + '</dd>' +
      '<dt>分析来源</dt><dd>' + a.model + '</dd></div>' +
      '<div class="section-title">色相分布（真实像素统计）</div>' + bars +
      (a.lowConfidence ? '<div class="diff-box" style="margin:8px 0 0">主色置信度偏低，画面可能不含明显安全帽或存在大面积高饱和背景，请在右侧人工复核帽色。</div>' : '') +
      '<p class="cr-detail" style="margin:8px 0 0">说明：颜色与候选框来自真实图片像素计算；<b>破损、佩戴方式、人脸结果</b>无法由色相统计可靠推断，需人工复核或接入真实 YOLOv8 服务。</p>';
  }

  /* ---------------- 复核表单 ---------------- */
  function readForm() {
    return {
      helmet: $('imgHelmet').checked, helmetOnHead: $('imgOnHead').checked,
      color: $('imgColor').value, damage: $('imgDamage').value, wear: $('imgWear').value,
      live: $('imgLive').checked, faceScore: parseFloat($('imgFace').value),
      staffId: $('imgStaff').value
    };
  }

  function effectiveColor(f) {
    if (f.color !== 'auto') return f.color;
    return st.analysis ? st.analysis.main : find(D.STAFF, 'id', f.staffId).color;
  }

  /* ---------------- 主流程 ---------------- */
  function run() {
    if (!st.image) { global.App.toast('请先导入图片或生成示例图片'); return; }
    var f = readForm(), staff = find(D.STAFF, 'id', f.staffId);
    var color = effectiveColor(f);
    var scene = E.resolveScene({
      id: 'IMG', label: '外部图片检测 · ' + st.image.name,
      desc: '图片来源 ' + st.image.name + '；颜色由' + (st.analysis ? st.analysis.model : '人工录入') + '得出，其余字段由' +
        (global.App.state.source ? '真实 YOLOv8 服务' : '操作员复核') + '填写',
      helmetPresent: f.helmet, color: color, damage: f.damage, wear: f.wear,
      live: f.live, faceScore: f.faceScore, expectedAllow: false
    }, staff);
    st.staff = staff; st.scene = scene;
    st.damageBoxes = f.damage !== 'intact' ? candidateDamageBox() : [];
    var useYolo = !!global.App.state.source;
    st.seed += 1;

    if (useYolo) {
      $('imgSourceChip').textContent = '推理源：真实 YOLOv8 服务';
      $('imgSourceChip').className = 'chip green';
      var input = {
        scene: scene, staff: staff, seed: st.seed, channel: global.App.state.channel,
        image_b64: st.image.rawDataUrl, view: st.view
      };
      E.infer(input, global.App.state.source).then(function (result) {
        finish(result, scene, staff, true);
      }).catch(function (err) {
        global.App.toast('模型服务不可用，已降级为像素分析 + 复核：' + err.message);
        useLocal(scene, staff, false);
      });
    } else {
      $('imgSourceChip').textContent = '推理源：像素分析 + 人工复核';
      $('imgSourceChip').className = 'chip amber';
      useLocal(scene, staff, false);
    }
  }

  function useLocal(scene, staff) {
    var result = E.buildResult({
      scene: scene, staff: staff, seed: st.seed, channel: global.App.state.channel,
      analysis: {
        hsvHue: st.analysis ? st.analysis.hsvHue : 0,
        clsProbs: st.analysis ? st.analysis.clsProbs : { white: .25, yellow: .25, blue: .25, red: .25 },
        conf: st.analysis ? st.analysis.conf : .5,
        model: st.analysis ? st.analysis.model + '（真实像素）' : '人工录入'
      },
      boxes: st.boxes, damageBoxes: st.damageBoxes,
      helmetOnHead: readForm().helmetOnHead,
      image: st.image, view: st.view, autoColor: $('imgColor').value === 'auto'
    });
    finish(result, scene, staff, false);
  }

  function finish(result, scene, staff, yolo) {
    st.result = result; st.decision = E.decide(result, {
      thresholds: global.App.state.thresholds, enabled: global.App.state.enabled
    });
    st.narration = E.narrate({ result: result, decision: st.decision, scene: scene, staff: staff });
    st.last = {
      result: result, decision: st.decision, narration: st.narration,
      scene: scene, staff: staff, imageOnly: true,
      img: {
        image: st.image, view: st.view, analysis: st.analysis,
        boxes: st.boxes, damageBoxes: st.damageBoxes, form: readForm()
      },
      cfg: { channel: global.App.state.channel, staffId: staff.id, sceneId: 'S01',
             overrides: null, thresholds: Object.assign({}, global.App.state.thresholds),
             enabled: Object.assign({}, global.App.state.enabled), seed: st.seed }
    };
    renderImage(); renderDecision(); renderNarrative(); renderCompare();
    global.App.toast(yolo ? '真实模型推理完成' : '图片检测完成（像素分析 + 复核）');
  }

  function renderDecision() {
    if (!st.decision) { $('imgDecision').innerHTML = '<p class="cr-detail" style="margin:0">等待检测。</p>'; return; }
    var rows = st.decision.checks.map(function (x) {
      var m = x.skipped ? '<span class="mark skip">—</span>' : (x.pass ? '<span class="mark ok">✔</span>' : '<span class="mark no">✘</span>');
      return '<div class="check-row"><div class="order">' + x.order + '</div><div>' +
        '<div class="cr-title">' + x.label + '</div><div class="cr-detail">' + x.detail + '</div></div>' + m + '</div>';
    }).join('');
    var d = st.decision;
    $('imgDecision').innerHTML = rows + '<div class="verdict ' + (d.allow ? 'allow' : 'deny') + '">' +
      '<div class="v-code">' + d.code + ' · ' + (E.CODE_TEXT[d.code] || '校验结果') + '</div>' +
      '<div class="v-text">' + (d.allow ? '五项校验全部通过，闸机放行' : '闸机保持关闭，拒绝通行') + '</div>' +
      '<div class="v-sub">' + d.summary + '</div></div>';
  }

  function renderNarrative() {
    if (!st.narration) { $('imgNarrative').innerHTML = '<p class="cr-detail" style="margin:0">等待检测。</p>'; return; }
    $('imgNarrative').innerHTML = st.narration.parts.map(function (p) {
      return '<div class="narrative-part"><div class="np-head">' + p.icon + ' · ' + p.title + '</div><div class="np-text">' + p.text + '</div></div>';
    }).join('');
  }

  function renderCompare() {
    if (!st.decision) { $('imgCompareNote').innerHTML = '执行一次图片检测后，可与场景预设做判定对比。'; return; }
    var sid = $('imgCompareScene').value;
    var base = find(D.SCENES, 'id', sid);
    var rs = E.resolveScene(base, st.staff);
    var r2 = E.infer({ scene: rs, staff: st.staff, seed: 777, channel: global.App.state.channel });
    var d2 = E.decide(r2, { thresholds: global.App.state.thresholds, enabled: global.App.state.enabled });
    var diff = [];
    st.decision.checks.forEach(function (x, i) {
      if (x.pass !== d2.checks[i].pass) diff.push(x.label);
    });
    $('imgCompareNote').innerHTML = '<b>图片结果 ' + st.decision.code + '</b> vs <b>预设「' + base.label + '」' + d2.code + '</b>：' +
      (diff.length ? '有 ' + diff.length + ' 项规则判定不同（' + diff.join('、') + '）。' : '五项规则判定完全一致。') +
      ' 预设真值期望' + (rs.expectedAllow ? '放行' : '拦截') + '，图片结果' + (st.decision.allow ? '放行' : '拦截') + '。' +
      ' 该对比可验证「同一员工在不同佩戴状态」下决策链的可解释性。';
  }

  /* ---------------- 台账 / 导出 / 重放 ---------------- */
  function saveToHistory() {
    if (!st.last) { global.App.toast('请先执行图片检测'); return; }
    global.App.addHistory(Object.assign({}, st.last));
    global.App.toast('已写入通行台账，可点击「复述重放」还原');
  }

  function exportReport() {
    if (!st.last) { global.App.toast('请先执行图片检测'); return; }
    var c = st.last, lines = [
      '工地入口安全帽佩戴检测报告（外部图片）',
      '生成时间：' + new Date().toLocaleString('zh-CN'),
      '图片：' + (c.img.image ? c.img.image.name : '-'),
      '员工：' + c.staff.name + '（' + c.staff.id + '） | 工种：' + c.staff.role,
      '推理源：' + (global.App.state.source ? '真实 YOLOv8 服务' : '浏览器端像素分析 + 人工复核'),
      '', c.narration.full, '', '--- 决策明细 ---'
    ];
    c.decision.checks.forEach(function (x) {
      lines.push(x.order + '. ' + x.label + '：' + (x.skipped ? '跳过' : x.pass ? '通过' : '未通过') + ' | ' + x.detail);
    });
    if (c.img.analysis) {
      lines.push('', '--- 像素分析 ---');
      lines.push('采样 ' + c.img.analysis.sampled + ' 点，主色 ' + D.COLORS[c.img.analysis.main].name +
        '，H≈' + c.img.analysis.hsvHue + '°，置信度 ' + fx(c.img.analysis.conf, 3));
    }
    var blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' });
    var a = document.createElement('a'); a.href = URL.createObjectURL(blob);
    a.download = 'helmet_image_report.txt'; a.click(); URL.revokeObjectURL(a.href);
    global.App.toast('报告已导出');
  }

  function copyNarrative() {
    if (!st.last) return;
    var txt = st.last.narration.full;
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(txt).then(function () { global.App.toast('复述已复制'); }, function () { global.App.toast('复制失败，请使用导出报告'); });
    } else global.App.toast('请使用导出报告');
  }

  function replay(ctx) {
    st.image = ctx.img.image; st.view = ctx.img.view; st.analysis = ctx.img.analysis;
    st.boxes = ctx.img.boxes; st.damageBoxes = ctx.img.damageBoxes;
    st.result = ctx.result; st.decision = ctx.decision; st.narration = ctx.narration;
    st.scene = ctx.scene; st.staff = ctx.staff; st.last = ctx;
    if (st.image) $('imgMeta').innerHTML = '<b>' + st.image.name + '</b><br>台账重放记录';
    renderImage(); renderAnalysis(); renderDecision(); renderNarrative(); renderCompare();
    global.App.toast('已重放图片检测复述');
  }

  /* ---------------- 初始化 ---------------- */
  function init() {
    fillSelect('imgStaff', D.STAFF.map(function (s) {
      return { value: s.id, label: s.id + ' · ' + s.name + '（授权' + D.COLORS[s.color].name + '帽）' };
    }), D.STAFF[0].id);
    fillSelect('imgColor', [{ value: 'auto', label: '采用像素分析主色（自动）' }].concat(
      Object.keys(D.COLORS).map(function (k) { return { value: k, label: D.COLORS[k].name + ' · ' + D.COLORS[k].grade }; })
    ), 'auto');
    fillSelect('imgDamage', Object.keys(D.DAMAGE_LABELS).map(function (k) {
      return { value: k, label: D.DAMAGE_LABELS[k] };
    }), 'intact');
    fillSelect('imgWear', Object.keys(D.WEAR_LABELS).map(function (k) {
      return { value: k, label: D.WEAR_LABELS[k] };
    }), 'correct');
    fillSelect('imgCompareScene', D.SCENES.map(function (s) {
      return { value: s.id, label: s.id + ' · ' + s.label };
    }), 'S01');

    var dz = $('dropZone'), fi = $('fileInput');
    dz.addEventListener('click', function () { fi.click(); });
    fi.addEventListener('change', function () { if (fi.files && fi.files[0]) loadFile(fi.files[0]); fi.value = ''; });
    ['dragenter', 'dragover'].forEach(function (t) {
      dz.addEventListener(t, function (e) { e.preventDefault(); dz.classList.add('over'); });
    });
    ['dragleave', 'drop'].forEach(function (t) {
      dz.addEventListener(t, function (e) { e.preventDefault(); dz.classList.remove('over'); });
    });
    dz.addEventListener('drop', function (e) {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0]) loadFile(e.dataTransfer.files[0]);
    });
    $('btnSample').addEventListener('click', makeSample);
    $('btnClearImage').addEventListener('click', function () {
      st.image = null; st.analysis = null; st.boxes = []; st.damageBoxes = [];
      st.result = st.decision = st.narration = st.last = null;
      $('imgMeta').innerHTML = '尚未导入图片';
      renderImage(); renderAnalysis(); renderDecision(); renderNarrative();
    });
    $('imgFace').addEventListener('input', function () {
      $('imgFaceVal').textContent = fx(this.value);
      $('imgFaceDup').textContent = fx(this.value);
    });
    $('btnAutoColor').addEventListener('click', function () {
      if (!st.analysis) { global.App.toast('请先导入图片'); return; }
      $('imgColor').value = st.analysis.main;
      global.App.toast('已采用像素分析主色：' + D.COLORS[st.analysis.main].name);
    });
    $('btnImgRun').addEventListener('click', run);
    $('imgShowBoxes').addEventListener('change', function () { st.showBoxes = this.checked; renderImage(); });
    $('imgShowGrid').addEventListener('change', function () { st.showGrid = this.checked; renderImage(); });
    $('imgCompareScene').addEventListener('change', renderCompare);
    $('btnImgSave').addEventListener('click', saveToHistory);
    $('btnImgExport').addEventListener('click', exportReport);
    $('btnImgCopy').addEventListener('click', copyNarrative);

    renderImage(); renderAnalysis(); renderDecision(); renderNarrative();
  }

  function fillSelect(id, items, value) {
    var el = $(id);
    el.innerHTML = items.map(function (it) {
      return '<option value="' + it.value + '">' + it.label + '</option>';
    }).join('');
    if (value !== undefined) el.value = value;
  }

  global.ImageLab = { init: init, run: run, replay: replay, analyze: analyze, state: st };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})(window);
