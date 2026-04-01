#!/bin/bash
# DeepAssist 서버 실행 스크립트 (Mac/Linux)
# 사용법: chmod +x start.sh && ./start.sh

echo "🚀 DeepAssist 시작 중..."

# FastAPI 파일 서버 (백그라운드)
echo "📁 FastAPI 파일 서버 시작 (포트 8000)..."
python3 server.py &
FASTAPI_PID=$!

# 잠시 대기 (FastAPI 기동 시간 확보)
sleep 2

# Streamlit UI
echo "🤖 Streamlit UI 시작 (포트 8501)..."
streamlit run app.py --server.port 8501 --server.address 0.0.0.0 &
STREAMLIT_PID=$!

echo ""
echo "✅ DeepAssist 실행 완료!"
echo "   📁 파일 서버 API: http://localhost:8000/docs"
echo "   🤖 웹 UI:        http://localhost:8501"
echo ""
echo "종료하려면 Ctrl+C를 누르세요."

# 종료 시 두 프로세스 모두 정리
trap "echo '🛑 종료 중...'; kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null; exit 0" INT TERM

# 프로세스 대기
wait
