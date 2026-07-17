// Tab 切换
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
  });
});

const API = '';

// ===== Tab 1: 数据集概览 =====
async function loadOverview() {
  const res = await fetch(`${API}/api/overview`);
  const d = await res.json();

  // 统计卡片
  const stats = [
    {label: '训练集样本', value: d.train_size.toLocaleString(), cls: 'blue'},
    {label: '测试集样本', value: d.test_size.toLocaleString(), cls: 'blue'},
    {label: '已知类数', value: d.n_known, cls: 'green'},
    {label: '未知攻击种数', value: d.n_unknown, cls: 'orange'},
    {label: '特征维度', value: d.feature_dim, cls: ''},
  ];
  document.getElementById('overview-stats').innerHTML = stats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value ${s.cls}">${s.value}</div></div>`
  ).join('');

  // 训练集分布柱状图
  const trainChart = echarts.init(document.getElementById('chart-train-dist'));
  const td = d.train_dist.sort((a, b) => b.n - a.n);
  trainChart.setOption({
    tooltip: { trigger: 'axis' },
    grid: { left: 50, right: 20, bottom: 90 },
    xAxis: { type: 'category', data: td.map(x => x.attack),
      axisLabel: { rotate: 45, color: '#86868b' }, axisLine: { lineStyle: { color: '#d2d2d7' } } },
    yAxis: { type: 'log', name: '样本数(log)', nameTextStyle: { color: '#86868b' },
      splitLine: { lineStyle: { color: '#e5e5ea' } }, axisLine: { show: false }, axisLabel: { color: '#86868b' } },
    series: [{ type: 'bar', data: td.map(x => x.n),
      itemStyle: { color: '#007aff', borderRadius: [4,4,0,0] } }],
  });

  // 未知攻击表
  const unk = d.unknown_attacks.sort((a, b) => b.n - a.n);
  document.getElementById('unknown-table').innerHTML = `
    <table>
      <thead><tr><th>未知攻击</th><th>样本数</th><th>是否数据极限</th></tr></thead>
      <tbody>${unk.map(u => `
        <tr>
          <td>${u.attack}</td>
          <td>${u.n}</td>
          <td>${u.is_overlap ? '<span class="tag overlap">重叠(无法检出)</span>' : '<span class="tag ok">可检出</span>'}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;

  // 数据极限列表
  document.getElementById('overlap-list').innerHTML =
    `<div class="unknown-grid">${d.overlap_attacks.map(a =>
      `<div class="unknown-item overlap"><span class="name">${a}</span></div>`).join('')}</div>`;
}

// ===== Tab 2: 批量评估 =====
async function loadTestFiles() {
  const res = await fetch(`${API}/api/test-files`);
  const files = await res.json();
  const sel = document.getElementById('test-file-select');
  sel.innerHTML = files.map(f => `<option value="${f.path}">${f.name}</option>`).join('');
}

document.getElementById('btn-evaluate').addEventListener('click', async () => {
  const btn = document.getElementById('btn-evaluate');
  const status = document.getElementById('eval-status');
  const sel = document.getElementById('test-file-select');
  btn.disabled = true;
  status.className = 'status running';
  status.textContent = '加载模型并评估中(约20秒)...';
  document.getElementById('eval-results').style.display = 'none';

  try {
    const res = await fetch(`${API}/api/evaluate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ test_file: sel.value }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || '评估失败');
    }
    const d = await res.json();
    renderEval(d);
    status.className = 'status done';
    status.textContent = `完成 · ${d.n_test} 条测试样本`;
  } catch (e) {
    status.className = 'status error';
    status.textContent = '失败 · ' + e.message;
    console.error(e);
  } finally {
    btn.disabled = false;
  }
});

function renderEval(d) {
  document.getElementById('eval-results').style.display = 'block';
  const u = d.unknown_detection, k = d.known_classification, o = d.overall;
  const stats = [
    {label: '未知检测 F1', value: u.F1, cls: 'green'},
    {label: '未知 P', value: u.P, cls: 'blue'},
    {label: '未知 R', value: u.R, cls: 'blue'},
    {label: '已知类 TNR', value: u.TNR, cls: 'green'},
    {label: '已知分类 acc', value: k.acc, cls: 'orange'},
    {label: '整体 acc', value: o.acc, cls: 'orange'},
  ];
  document.getElementById('eval-stats').innerHTML = stats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value ${s.cls}">${s.value}</div></div>`
  ).join('');

  // 未知攻击检出率柱状图
  const unkChart = echarts.init(document.getElementById('chart-unknown-detect'));
  const unk = d.per_unknown.sort((a, b) => b.detect_rate - a.detect_rate);
  unkChart.setOption({
    tooltip: { trigger: 'axis', formatter: p => `${p[0].name}<br/>检出率: ${(p[0].value*100).toFixed(1)}%<br/>n=${p[0].data.n}` },
    grid: { left: 50, right: 20, bottom: 90 },
    xAxis: { type: 'category', data: unk.map(x => x.attack),
      axisLabel: { rotate: 45, color: '#86868b' }, axisLine: { lineStyle: { color: '#d2d2d7' } } },
    yAxis: { type: 'value', max: 1, name: '检出率', nameTextStyle: { color: '#86868b' },
      splitLine: { lineStyle: { color: '#e5e5ea' } }, axisLine: { show: false }, axisLabel: { color: '#86868b' } },
    series: [{
      type: 'bar',
      data: unk.map(x => ({ value: x.detect_rate, n: x.n })),
      itemStyle: { color: (p) => p.value < 0.1 ? '#ff3b30' : (p.value < 0.5 ? '#ff9500' : '#34c759'),
        borderRadius: [4,4,0,0] },
    }],
  });

  // 已知类表格
  const known = d.per_known;
  document.getElementById('known-table').innerHTML = `
    <table>
      <thead><tr><th>已知类</th><th>n</th><th>召回</th><th>接受率</th><th>精确率</th><th>F1</th></tr></thead>
      <tbody>${known.map(k => `
        <tr>
          <td>${k.attack}</td><td>${k.n}</td>
          <td>${k.n ? k.rec : '—'}</td>
          <td>${k.n ? k.accept : '—'}</td>
          <td>${k.n ? k.prec : '—'}</td>
          <td>${k.n ? k.f1 : '—'}</td>
        </tr>`).join('')}
      </tbody>
    </table>`;

  // 混淆矩阵
  const cmChart = echarts.init(document.getElementById('chart-confusion'));
  const labels = d.labels;
  const data = [];
  for (let i = 0; i < d.confusion_matrix.length; i++)
    for (let j = 0; j < d.confusion_matrix[i].length; j++)
      data.push([j, i, d.confusion_matrix[i][j]]);
  const maxv = Math.max(...data.map(x => x[2]));
  cmChart.setOption({
    tooltip: { position: 'top', formatter: p => `真:${labels[p.value[1]]}<br/>预:${labels[p.value[0]]}<br/>${p.value[2]}` },
    grid: { left: 120, right: 20, bottom: 120 },
    xAxis: { type: 'category', data: labels, name: '预测',
      axisLabel: { rotate: 60, color: '#86868b' }, axisLine: { lineStyle: { color: '#d2d2d7' } } },
    yAxis: { type: 'category', data: labels, name: '真实',
      axisLabel: { rotate: 30, color: '#86868b' }, axisLine: { lineStyle: { color: '#d2d2d7' } } },
    visualMap: { min: 0, max: maxv, calculable: true, orient: 'horizontal', left: 'center', bottom: 0,
      inRange: { color: ['#f5f5f7', '#a8c7ff', '#007aff'] },
      textStyle: { color: '#86868b' } },
    series: [{ type: 'heatmap', data: data,
      label: { show: false },
      emphasis: { itemStyle: { shadowBlur: 10 } } }],
  });
}

// ===== Tab 3: 模型方案 =====
async function loadModel() {
  const res = await fetch(`${API}/api/model`);
  const d = await res.json();

  document.getElementById('model-title').textContent = d.title;
  const stats = [
    {label: '输入维度', value: d.in_dim, cls: ''},
    {label: '已知类数', value: d.n_classes, cls: 'green'},
    {label: '分类器参数量', value: d.clf_params.toLocaleString(), cls: 'blue'},
    {label: '自编码器参数量', value: d.ae_params.toLocaleString(), cls: 'blue'},
    {label: '未知检测 F1', value: d.results.unknown_f1, cls: 'green'},
    {label: '已知类 acc', value: d.results.known_acc, cls: 'orange'},
  ];
  document.getElementById('model-stats').innerHTML = stats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value ${s.cls}">${s.value}</div></div>`
  ).join('');

  // 架构图 (CSS 流程图)
  document.getElementById('arch-diagram').innerHTML = `
    <div class="arch-flow">
      <div class="arch-node input">输入样本<br><small>41 特征 → ${d.in_dim}维预处理</small></div>
      <div class="arch-arrow">↓</div>
      <div class="arch-branch">
        <div class="arch-col">
          <div class="arch-node clf">MLP 分类器</div>
          <div class="arch-arrow">↓</div>
          <div class="arch-node out">23 类 logits<br><small>+ softmax 最大概率</small></div>
        </div>
        <div class="arch-col">
          <div class="arch-node ae">自编码器</div>
          <div class="arch-arrow">↓</div>
          <div class="arch-node out">重构误差<br><small>(OOD 主信号)</small></div>
        </div>
      </div>
      <div class="arch-arrow">↓</div>
      <div class="arch-node fusion">融合: 0.818×AE误差 + 0.182×(1−softmax)</div>
      <div class="arch-arrow">↓</div>
      <div class="arch-node decision">
        <span class="branch-yes">score &gt; 阈值 → <b>unknown</b></span>
        <span class="branch-no">否则 → <b>已知类 argmax</b></span>
      </div>
    </div>`;

  // 分类器层
  document.getElementById('clf-layers').innerHTML = renderLayers(d.classifier.layers);
  document.getElementById('clf-purpose').innerHTML =
    `<b>损失:</b> ${d.classifier.loss}<br><b>作用:</b> ${d.classifier.purpose}`;

  // AE 层
  document.getElementById('ae-layers').innerHTML = renderLayers(d.autoencoder.layers);
  document.getElementById('ae-purpose').innerHTML =
    `<b>损失:</b> ${d.autoencoder.loss}<br><b>作用:</b> ${d.autoencoder.purpose}`;

  // OOD 融合
  document.getElementById('ood-fusion').innerHTML = `
    <div class="formula">${d.ood_fusion.formula}</div>
    <p class="hint"><b>判定规则:</b> ${d.ood_fusion.rule}</p>
    <p class="hint"><b>阈值:</b> ${d.ood_fusion.threshold}</p>`;

  // 为什么 AE
  document.getElementById('why-ae').innerHTML = d.why_ae.map(r => `<li>${r}</li>`).join('');

  // 关键决策
  document.getElementById('key-decisions').innerHTML = d.key_decisions.map(kd => `
    <div class="decision-item">
      <div class="decision-k">${kd.k}</div>
      <div class="decision-v">${kd.v}</div>
    </div>`).join('');

  // 训练配置
  const t = d.training;
  const tstats = [
    {label: '分类器 epoch', value: t.epochs_cls},
    {label: 'AE epoch', value: t.epochs_ae},
    {label: '分类器 lr', value: t.lr_cls},
    {label: 'AE lr', value: t.lr_ae},
    {label: 'batch size', value: t.batch},
    {label: '随机种子', value: t.seed},
  ];
  document.getElementById('training-config').innerHTML = tstats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value">${s.value}</div></div>`
  ).join('');
}

function renderLayers(layers) {
  return layers.map((l, i) => `
    <div class="layer-item">
      <div class="layer-idx">${i + 1}</div>
      <div class="layer-body">
        <div class="layer-name">${l.name}</div>
        <div class="layer-shape">${l.shape}</div>
      </div>
    </div>`).join('');
}

// ===== Tab: 训练 =====
const TRAIN_STAGES = [
  "加载数据", "训练分类器", "训练自编码器", "保存模型",
  "提取激活向量", "拟合", "评估", "完成",
];
let _trainTimer = null;
let _trainUnkChart = null;

function stageFraction(stage) {
  const i = TRAIN_STAGES.findIndex(s => stage && stage.includes(s));
  if (i < 0) return 0.05;
  return Math.min(1, (i + 1) / TRAIN_STAGES.length);
}

function renderTrainMetrics(d) {
  document.getElementById('train-results').style.display = 'block';
  const u = d.unknown_detection, k = d.known_classification, o = d.overall;
  const stats = [
    {label: '未知检测 F1', value: u.F1, cls: 'green'},
    {label: '未知 P', value: u.P, cls: 'blue'},
    {label: '未知 R', value: u.R, cls: 'blue'},
    {label: '已知类 TNR', value: u.TNR, cls: 'green'},
    {label: '已知分类 acc', value: k.acc, cls: 'orange'},
    {label: '整体 acc', value: o.acc, cls: 'orange'},
  ];
  document.getElementById('train-stats').innerHTML = stats.map(s =>
    `<div class="stat-item"><div class="label">${s.label}</div><div class="value ${s.cls}">${s.value}</div></div>`
  ).join('');

  const el = document.getElementById('chart-train-unknown');
  if (_trainUnkChart) _trainUnkChart.dispose();
  _trainUnkChart = echarts.init(el);
  const unk = d.per_unknown.sort((a, b) => b.detect_rate - a.detect_rate);
  _trainUnkChart.setOption({
    tooltip: { trigger: 'axis', formatter: p => `${p[0].name}<br/>检出率: ${(p[0].value*100).toFixed(1)}%<br/>n=${p[0].data.n}` },
    grid: { left: 50, right: 20, bottom: 90 },
    xAxis: { type: 'category', data: unk.map(x => x.attack),
      axisLabel: { rotate: 45, color: '#86868b' }, axisLine: { lineStyle: { color: '#d2d2d7' } } },
    yAxis: { type: 'value', max: 1, name: '检出率', nameTextStyle: { color: '#86868b' },
      splitLine: { lineStyle: { color: '#e5e5ea' } }, axisLine: { show: false }, axisLabel: { color: '#86868b' } },
    series: [{ type: 'bar', data: unk.map(x => ({ value: x.detect_rate, n: x.n })),
      itemStyle: { color: (p) => p.value < 0.1 ? '#ff3b30' : (p.value < 0.5 ? '#ff9500' : '#34c759'),
        borderRadius: [4,4,0,0] } }],
  });
}

async function pollTrain() {
  try {
    const res = await fetch(`${API}/api/train/status`);
    const d = await res.json();
    const log = document.getElementById('train-log');
    log.textContent = d.logs || '(等待输出...)';
    log.scrollTop = log.scrollHeight;
    const stageEl = document.getElementById('train-stage');
    stageEl.textContent = d.stage || '';
    document.getElementById('train-elapsed').textContent =
      d.elapsed != null ? `耗时 ${d.elapsed}s` : '';
    document.getElementById('train-meter-fill').style.width =
      (stageFraction(d.stage) * 100).toFixed(0) + '%';

    const status = document.getElementById('train-status');
    if (d.running) {
      status.className = 'status running';
      status.textContent = `训练中 · ${d.stage || ''}`;
    } else if (d.error) {
      status.className = 'status error';
      status.textContent = '训练失败';
      log.textContent = d.error;
      _trainTimer = null;
      return;
    } else if (d.done) {
      status.className = 'status done';
      status.textContent = '训练完成';
      document.getElementById('train-meter-fill').style.width = '100%';
      if (d.metrics) renderTrainMetrics(d.metrics);
      _trainTimer = null;
      return;
    }
    _trainTimer = setTimeout(pollTrain, 1000);
  } catch (e) {
    console.error(e);
    _trainTimer = setTimeout(pollTrain, 2000);
  }
}

document.getElementById('btn-train').addEventListener('click', async () => {
  const btn = document.getElementById('btn-train');
  const status = document.getElementById('train-status');
  const cfg = {
    epochs_cls: parseInt(document.getElementById('cfg-epochs-cls').value) || 40,
    epochs_ae: parseInt(document.getElementById('cfg-epochs-ae').value) || 30,
    oversample: document.getElementById('cfg-oversample').checked,
    full_ood: document.getElementById('cfg-full-ood').checked,
  };
  btn.disabled = true;
  status.className = 'status running';
  status.textContent = '启动训练...';
  document.getElementById('train-progress').style.display = 'block';
  document.getElementById('train-results').style.display = 'none';
  document.getElementById('train-log').textContent = '';
  document.getElementById('train-meter-fill').style.width = '0%';
  try {
    const res = await fetch(`${API}/api/train`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || '启动失败');
    }
    if (_trainTimer) clearTimeout(_trainTimer);
    pollTrain();
  } catch (e) {
    status.className = 'status error';
    status.textContent = '失败 · ' + e.message;
  } finally {
    btn.disabled = false;
  }
});

// 页面加载时检查是否有进行中的训练
async function checkOngoingTrain() {
  try {
    const res = await fetch(`${API}/api/train/status`);
    const d = await res.json();
    if (d.running) {
      document.getElementById('train-progress').style.display = 'block';
      document.getElementById('btn-train').disabled = true;
      pollTrain();
    } else if (d.done && d.metrics) {
      document.getElementById('train-progress').style.display = 'block';
      document.getElementById('train-meter-fill').style.width = '100%';
      document.getElementById('train-stage').textContent = '完成';
      renderTrainMetrics(d.metrics);
    }
  } catch (e) {}
}

// 初始化
loadOverview();
loadTestFiles();
loadModel();
checkOngoingTrain();
