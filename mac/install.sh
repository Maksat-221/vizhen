#!/bin/bash
# Установка Vision (Вижен) на Mac одной командой:
#   curl -fsSL https://maksat-221.github.io/vizhen/mac/install.sh | bash
set -e
DIR="$HOME/Vision"
BASE="https://maksat-221.github.io/vizhen/mac"

echo ""
echo "  ▶ Устанавливаю Vision в $DIR"
mkdir -p "$DIR"
curl -fsSL "$BASE/vision.py?$(date +%s)" -o "$DIR/vision.py"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null; then
  echo ""
  echo "  ✖ На Mac нет Python 3. Сейчас откроется окно установки инструментов Apple — нажми «Установить»."
  echo "    Когда закончится, запусти эту же команду ещё раз."
  xcode-select --install 2>/dev/null || true
  exit 1
fi

echo "  ▶ Ставлю библиотеки (минута-две)…"
[ -d "$DIR/venv" ] || python3 -m venv "$DIR/venv"
"$DIR/venv/bin/python" -m pip install -q --upgrade pip
"$DIR/venv/bin/python" -m pip install -q sounddevice numpy requests

cat > "$DIR/Vision.command" <<EOF
#!/bin/bash
cd "$DIR" && exec "$DIR/venv/bin/python" "$DIR/vision.py" "\$@"
EOF
chmod +x "$DIR/Vision.command"
ln -sf "$DIR/Vision.command" "$HOME/Desktop/Vision.command" 2>/dev/null || true

echo ""
echo "  ✔ Готово. Ярлык Vision.command лежит на Рабочем столе — двойной щелчок запускает ассистента."
echo "  ▶ Первый запуск: вставь ключи Anthropic и OpenAI, потом разреши Терминалу доступ к микрофону."
echo ""
exec "$DIR/venv/bin/python" "$DIR/vision.py" < /dev/tty
