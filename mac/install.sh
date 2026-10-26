#!/bin/bash
# Установка Vision (Вижен) на Mac одной командой:
#   curl -fsSL https://maksat-221.github.io/vizhen/mac/install.sh | bash
set -e
DIR="$HOME/Vision"
BASE="https://maksat-221.github.io/vizhen/mac"

echo ""
echo "  ▶ Устанавливаю Vision в $DIR"
mkdir -p "$DIR"
curl -fsSL "$BASE/vision.py?$(date +%s)" -o "$DIR/vision.py" 2>/dev/null || curl -fsSL "https://raw.githubusercontent.com/Maksat-221/vizhen/main/mac/vision.py" -o "$DIR/vision.py"

# Python ставим сами через uv: не нужны ни инструменты Apple, ни пароль администратора
export PATH="$HOME/.local/bin:$PATH"
export UV_PYTHON_PREFERENCE=only-managed
if ! command -v uv >/dev/null 2>&1; then
  echo "  ▶ Скачиваю установщик Python (uv)…"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "  ▶ Ставлю Python и библиотеки (1–3 минуты)…"
[ -x "$DIR/venv/bin/python" ] || uv venv -q --python 3.12 "$DIR/venv"
uv pip install -q --python "$DIR/venv/bin/python" sounddevice numpy requests edge-tts
echo "  ▶ Ставлю бесплатное распознавание речи (Whisper на этом Mac)…"
uv pip install -q --python "$DIR/venv/bin/python" mlx-whisper || echo "  (Whisper не встал — будет распознавание через Gemini)"

cat > "$DIR/Vision.command" <<EOF
#!/bin/bash
cd "$DIR" && exec "$DIR/venv/bin/python" "$DIR/vision.py" "\$@"
EOF
chmod +x "$DIR/Vision.command"
ln -sf "$DIR/Vision.command" "$HOME/Desktop/Vision.command" 2>/dev/null || true

echo ""
echo "  ✔ Готово. Ярлык Vision.command лежит на Рабочем столе — двойной щелчок запускает ассистента."
echo "  ▶ Первый запуск: вставь бесплатный ключ Gemini (aistudio.google.com/apikey), потом разреши Терминалу микрофон."
echo ""
exec "$DIR/venv/bin/python" "$DIR/vision.py" < /dev/tty
