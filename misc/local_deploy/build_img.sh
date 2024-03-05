# Usage: ./build_img.sh
# build docker image local_deploy_img from Dockerfile. If the image already exists, it will be removed and rebuilt.

docker build --no-cache -t local_deploy_img -f Dockerfile .

