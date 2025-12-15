set -e

ACTION="$1"

case "$ACTION" in
  git)
    MSG="${2:-update}"
    git add .
    git commit -m "$MSG"
    git push
    ;;
  update)
    # 调用更新脚本
    python "e:/note/test.py"
    ;;
  server)
    # 启动本地服务器
    teedoc serve
    ;;
  *)
    echo "用法:"
    echo "  $0 git \"commit message\"   # 执行 git 提交与推送"
    echo "  $0 update                   # 扫描 doc 并更新配置"
    echo "  $0 server                   # 开启teedoc本地服务器"
    exit 1
    ;;
esac