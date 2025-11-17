// app/js/script2.js
import { predictFile, healthCheck } from "../../shared/js/api.js";

// ====== 부팅 확인 로그 (콘솔에서 꼭 보세요) ======
console.log("[app] boot ok");

// ====== 공통 유틸 ======
const $ = (s) => document.querySelector(s);
const views = ["view-idle", "view-loading", "view-ok", "view-bad", "view-fail"];
function show(id) {
  views.forEach((v) => {
    const el = document.getElementById(v);
    if (el) el.classList.toggle("hidden", v !== id);
  });
}
const wait = (ms) => new Promise((r) => setTimeout(r, ms));

// ====== 요소 ======
// (중복 선언 절대 금지!)
const fileInput     = $("#fileInput");
const dropZone      = $("#dropZone");
const btnAnalyze    = $("#btnAnalyze");

const btnAnotherOk  = $("#btnAnotherOk");
const btnReportOk   = $("#btnReportOk");
const btnAnotherBad = $("#btnAnotherBad");
const btnReportBad  = $("#btnReportBad");
const btnRetry      = $("#btnRetry");
const btnGoIdle     = $("#btnGoIdle");

// 시뮬레이터 버튼 (HTML에 없으면 null이어도 OK)
const btnSimOk      = $("#btnSimOk");
const btnSimBad     = $("#btnSimBad");
const btnSimFail    = $("#btnSimFail");

// 헤더 버튼
const btnLogin      = $("#btnLogin");
const btnSignup     = $("#btnSignup");

// API 테스트 섹션
const btnPredict    = $("#btnPredict");
const btnHealth     = $("#btnHealth");
const predictResult = $("#predictResult");
const healthResult  = $("#healthResult");

// ====== 방어: 필수 요소 점검 ======
if (!fileInput || !btnAnalyze) {
  console.warn("[app] 필수 요소 누락", { fileInput: !!fileInput, btnAnalyze: !!btnAnalyze });
}

// ====== 로그인/회원가입 리스너 ======
btnLogin?.addEventListener("click", () => {
  alert("Login 버튼 클릭됨 (추후 실제 로그인 연동)");
});
btnSignup?.addEventListener("click", () => {
  alert("Sign up 버튼 클릭됨 (추후 회원가입 연동)");
});

// ====== 업로드/분석 플로우 ======
btnAnalyze?.addEventListener("click", async () => {
  const file = fileInput?.files?.[0];
  if (!file) { alert("파일을 선택해 주세요."); return; }
  await analyzeFile(file);
});

async function analyzeFile(file) {
  show("view-loading");
  try {
    // 실제 API 연동 시:
    // const data = await predictFile(file);
    // if (data?.is_malware) {
    //   $("#bad-reasons").innerHTML = `• ${data.predicted_label} (${(data.predicted_score*100).toFixed(1)}%)`;
    //   show("view-bad");
    // } else {
    //   $("#ok-hash").textContent = data.sha256 || "-";
    //   $("#ok-time").textContent = (data.elapsed_sec ?? 0).toString();
    //   show("view-ok");
    // }
    // return;

    // 데모: 1.2초 대기 후 랜덤 분기
    await wait(1200);
    const mock = Math.random();
    if (mock < 0.33) {
      $("#ok-hash").textContent = "(demo) 1234...ABCD";
      $("#ok-time").textContent = "1.2";
      show("view-ok");
    } else if (mock < 0.66) {
      $("#bad-reasons").innerHTML = "• (demo) 의심 API 다수 사용<br>• (demo) 비정상 트래픽";
      show("view-bad");
    } else {
      $("#fail-msg").textContent = "네트워크 오류 또는 분석 타임아웃(데모).";
      show("view-fail");
    }
  } catch (e) {
    $("#fail-msg").textContent = "예상치 못한 오류가 발생했습니다.";
    show("view-fail");
  }
}

// 결과 화면 버튼들
btnAnotherOk?.addEventListener("click", () => show("view-idle"));
btnReportOk?.addEventListener("click", () => alert("리포트 저장(추후 구현)"));
btnAnotherBad?.addEventListener("click", () => show("view-idle"));
btnReportBad?.addEventListener("click", () => alert("리포트 저장(추후 구현)"));
btnRetry?.addEventListener("click", () => { show("view-loading"); setTimeout(() => show("view-idle"), 600); });
btnGoIdle?.addEventListener("click", () => show("view-idle"));

// ====== 드래그 앤 드롭 ======
["dragenter", "dragover"].forEach((evt) =>
  dropZone?.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone?.addEventListener(evt, (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); })
);
dropZone?.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  if (dt?.files?.length) fileInput.files = dt.files;
});

// ====== 시뮬레이터 ======
function simulate(type) {
  show("view-loading");
  setTimeout(() => {
    if (type === "ok") {
      $("#ok-hash").textContent = "(demo) DEAD-BEEF-1234";
      $("#ok-time").textContent = "0.8";
      show("view-ok");
    } else if (type === "bad") {
      $("#bad-reasons").innerHTML = "• (demo) 악성 패턴 코드 감지<br>• (demo) 의심 네트워크 호출";
      show("view-bad");
    } else {
      $("#fail-msg").textContent = "분석 서버 응답 실패(시뮬레이션).";
      show("view-fail");
    }
  }, 800);
}

// (안전) 위임 + 캡처 단계로 어떤 경우에도 클릭 감지
document.addEventListener("click", (e) => {
  const t = e.target;
  if (!(t instanceof Element)) return;
  if (t.id === "btnSimOk")   { e.preventDefault(); simulate("ok");  }
  if (t.id === "btnSimBad")  { e.preventDefault(); simulate("bad"); }
  if (t.id === "btnSimFail") { e.preventDefault(); simulate("fail");}
}, true);

// ====== API 테스트 ======
btnPredict?.addEventListener("click", async () => {
  const file = fileInput?.files?.[0];
  if (!file) { predictResult.textContent = "파일을 선택하세요."; return; }
  predictResult.textContent = "요청 중...";
  try {
    const data = await predictFile(file);
    predictResult.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    predictResult.textContent = `API 오류: ${String(err?.message || err)}`;
  }
});

btnHealth?.addEventListener("click", async () => {
  healthResult.textContent = "확인 중...";
  try {
    const data = await healthCheck();
    healthResult.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    healthResult.textContent = `Health 오류: ${String(err?.message || err)}`;
  }
});

// ====== 전역 에러 헨들링(진단용) ======
window.addEventListener("error", (e) => {
  console.error("[global error]", e.message, e.filename, e.lineno);
});
window.addEventListener("unhandledrejection", (e) => {
  console.error("[promise rejection]", e.reason);
});
