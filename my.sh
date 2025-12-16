set -e

ACTION="$1"

do_update() {
  python "./update.py"
}

case "$ACTION" in
  git)
    MSG="${2:-update}"

    # 先自动执行 update
    if ! do_update; then
      echo "更新脚本执行失败，终止 git 提交"
      exit 1
    fi

    git add .
    if ! git commit -m "$MSG"; then
      echo "Git 提交失败"
      exit 1
    fi
    git push
    ;;
  update)
    if ! do_update; then
      echo "更新脚本执行失败"
      exit 1
    fi
    ;;
  server)
    teedoc serve
    ;;
  *)
    echo "用法:"
    echo "  $0 git \"commit message\"   # 执行 git 提交与推送（会先执行 update）"
    echo "  $0 update                   # 扫描 doc 并更新配置"
    echo "  $0 server                   # 开启 teedoc 本地服务器"
    exit 1
    ;;
esac