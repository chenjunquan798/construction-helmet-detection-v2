/* 工地入口安全帽检测系统 - 基础数据
 * 颜色等级、人员库、场景预设、决策规则定义
 * 注意：颜色与工种的映射为项目可配置项，不写死在模型中
 */
(function (global) {
  'use strict';

  var COLORS = {
    white:  { key: 'white',  name: '白色', grade: '管理 / 监理',  hex: '#f4f5f7', stroke: '#9aa3ad', text: '#2c2c2a', hue: 0,   desc: '项目经理、监理、甲方管理人员' },
    yellow: { key: 'yellow', name: '黄色', grade: '普通施工人员', hex: '#f5c518', stroke: '#b58900', text: '#3d2f00', hue: 48,  desc: '土建、普工、一般作业人员' },
    blue:   { key: 'blue',   name: '蓝色', grade: '技术 / 特种作业', hex: '#2f7fd1', stroke: '#1a5a9c', text: '#ffffff', hue: 210, desc: '电工、焊工、机械操作、技术人员' },
    red:    { key: 'red',    name: '红色', grade: '安全 / 消防 / 访客', hex: '#d9483f', stroke: '#a32d2d', text: '#ffffff', hue: 4,   desc: '安全员、消防、外来访客' }
  };

  var STAFF = [
    { id: 'G-0231', name: '张建国', dept: '土建一队', role: '架子工',       color: 'yellow', baseScore: 0.94 },
    { id: 'G-0188', name: '李慧敏', dept: '项目部',   role: '安全总监',     color: 'red',    baseScore: 0.96 },
    { id: 'G-0402', name: '王技术', dept: '机电组',   role: '特种作业电工', color: 'blue',   baseScore: 0.92 },
    { id: 'G-0057', name: '陈监理', dept: '监理办',   role: '总监理',       color: 'white',  baseScore: 0.95 },
    { id: 'G-0316', name: '赵小工', dept: '土建二队', role: '普工',         color: 'yellow', baseScore: 0.89 },
    { id: 'G-0274', name: '周焊接', dept: '钢结构',   role: '焊工',         color: 'blue',   baseScore: 0.91 }
  ];

  var DAMAGE_LABELS = {
    intact: '完好', crack: '裂纹', dent: '凹陷', chip: '帽檐缺口', aging: '老化脆化'
  };

  var WEAR_LABELS = {
    correct:    '规范佩戴',
    unstrapped: '未系下颚带',
    backward:   '安全帽反戴',
    handheld:   '手持 / 未戴在头上',
    tilted:     '严重倾斜'
  };

  /* 场景预设：color 为 null 表示跟随该员工授权颜色（合规） */
  var SCENES = [
    { id: 'S01', label: '合规通行',           tag: '正常', helmetPresent: true,  color: null,    damage: 'intact', wear: 'correct',    live: true,  faceScore: 0.95, expectedAllow: true,  desc: '佩戴本人授权颜色、盔体完好、下颚带系紧，人脸活体通过' },
    { id: 'S02', label: '未佩戴安全帽',       tag: '违规', helmetPresent: false, color: null,    damage: 'intact', wear: 'correct',    live: true,  faceScore: 0.93, expectedAllow: false, desc: '头部无帽，检测不到盔体目标' },
    { id: 'S03', label: '盔体裂纹',           tag: '违规', helmetPresent: true,  color: null,    damage: 'crack',  wear: 'correct',    live: true,  faceScore: 0.94, expectedAllow: false, desc: '顶部视角发现贯穿性裂纹，破损分支高置信度告警' },
    { id: 'S04', label: '帽色与等级不符',     tag: '违规', helmetPresent: true,  color: 'red',   damage: 'intact', wear: 'correct',    live: true,  faceScore: 0.93, expectedAllow: false, desc: '普工佩戴红色安全帽，越权佩戴' },
    { id: 'S05', label: '未系下颚带',         tag: '违规', helmetPresent: true,  color: null,    damage: 'intact', wear: 'unstrapped', live: true,  faceScore: 0.92, expectedAllow: false, desc: '帽在头上但下颚带垂落，几何规则判定不规范' },
    { id: 'S06', label: '安全帽反戴',         tag: '违规', helmetPresent: true,  color: null,    damage: 'intact', wear: 'backward',   live: true,  faceScore: 0.91, expectedAllow: false, desc: '帽檐朝后，关键点与帽前沿方向不一致' },
    { id: 'S07', label: '手持安全帽',         tag: '违规', helmetPresent: true,  color: null,    damage: 'intact', wear: 'handheld',   live: true,  faceScore: 0.94, expectedAllow: false, desc: '检测到帽但不在头顶区域，属于无效佩戴' },
    { id: 'S08', label: '人脸活体失败',       tag: '异常', helmetPresent: true,  color: null,    damage: 'intact', wear: 'correct',    live: false, faceScore: 0.41, expectedAllow: false, desc: '照片 / 视频翻拍攻击，活体检测拦截' },
    { id: 'S09', label: '盔体严重凹陷',       tag: '违规', helmetPresent: true,  color: null,    damage: 'dent',   wear: 'correct',    live: true,  faceScore: 0.93, expectedAllow: false, desc: '盔体受冲击变形，结构强度不达标' },
    { id: 'S10', label: '安全帽严重倾斜',     tag: '违规', helmetPresent: true,  color: null,    damage: 'intact', wear: 'tilted',     live: true,  faceScore: 0.92, expectedAllow: false, desc: '帽体覆盖头顶比例不足，左右偏移超阈值' },
    { id: 'S11', label: '帽檐缺口',           tag: '违规', helmetPresent: true,  color: null,    damage: 'chip',   wear: 'correct',    live: true,  faceScore: 0.94, expectedAllow: false, desc: '侧面视角发现帽檐缺损' },
    { id: 'S12', label: '低相似度 + 完好帽',  tag: '边界', helmetPresent: true,  color: null,    damage: 'intact', wear: 'correct',    live: true,  faceScore: 0.68, expectedAllow: false, desc: '人脸相似度处于阈值附近，用于阈值灵敏度对比' }
  ];

  var RULES = [
    { id: 'face',   label: '人脸活体与 1:N 比对', order: 1, hint: 'liveness + face.score' },
    { id: 'helmet', label: '安全帽存在',         order: 2, hint: 'helmet.conf' },
    { id: 'wear',   label: '佩戴规范',           order: 3, hint: 'keypoints + geometry' },
    { id: 'damage', label: '盔体完好',           order: 4, hint: 'damage.prob' },
    { id: 'color',  label: '帽色与授权等级一致', order: 5, hint: 'color.predicted' }
  ];

  var DEFAULT_THRESHOLDS = {
    faceScore: 0.70,
    detConf: 0.50,
    wearConf: 0.60,
    damageProb: 0.50,
    colorConf: 0.60
  };

  global.SITE_DATA = {
    COLORS: COLORS,
    STAFF: STAFF,
    SCENES: SCENES,
    RULES: RULES,
    DAMAGE_LABELS: DAMAGE_LABELS,
    WEAR_LABELS: WEAR_LABELS,
    DEFAULT_THRESHOLDS: DEFAULT_THRESHOLDS,
    CHANNELS: ['A-01 员工入口', 'A-02 员工入口', 'B-01 访客通道', 'B-02 应急通道']
  };
})(window);
