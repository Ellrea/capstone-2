// PROJECT_ROOT/shared/js/api.js

export let BASE_URL = "https://capstone-2-malware.onrender.com"; // 예: "http://127.0.0.1:8000" 또는 "https://your-backend.com"

export function setBaseURL(url) {
  BASE_URL = (url || "").trim();
}

export const endpoints = {
  health: "/health",
  predict: "/predict",
};

function buildURL(path) {
  if (!BASE_URL) {
    throw new Error("BASE_URL이 설정되지 않았습니다. shared/js/api.js의 BASE_URL을 먼저 지정하세요.");
  }
  const base = BASE_URL.replace(/\/$/, "");
  const p = String(path || "").replace(/^\//, "");
  return `${base}/${p}`;
}

// 공통 요청 헬퍼
async function request(path, options = {}) {
  const res = await fetch(buildURL(path), options);

  const contentType = res.headers.get("content-type") || "";
  const isJSON = contentType.includes("application/json");

  let data;
  try {
    data = isJSON ? await res.json() : await res.text();
  } catch {
    data = null;
  }

  if (!res.ok) {
    const msg =
      (data && (data.detail || data.message)) ||
      (typeof data === "string" ? data : "") ||
      res.statusText ||
      "요청 실패";
    throw new Error(`HTTP ${res.status} - ${msg}`);
  }

  return data;
}

// ====== 공개 API 함수들 ======

// 헬스체크: GET /health
export async function healthCheck() {
  return request(endpoints.health, { method: "GET" });
}

// 파일 예측: POST /predict (FormData: file)
export async function predictFile(file, fieldName = "file") {
  const fd = new FormData();
  fd.append(fieldName, file);
  return request(endpoints.predict, { method: "POST", body: fd });
}

// 선택: JSON POST/GET 유틸(추가 엔드포인트 테스트용)
export async function postJSON(path, payload) {
  return request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload ?? {}),
  });
}

export async function getJSON(path, query = {}) {
  const qs = new URLSearchParams(query).toString();
  const url = qs ? `${path}?${qs}` : path;
  return request(url, { method: "GET" });
}
