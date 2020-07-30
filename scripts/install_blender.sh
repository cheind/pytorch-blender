set -e

NAME="blender-2.83.3-linux64"
NAMETAR="${NAME}.tar.xz"
CACHE="${HOME}/.blender-cache"
TAR="${CACHE}/${NAMETAR}"
URL="http://mirror.cs.umn.edu/blender.org/release/Blender2.83/${NAMETAR}"

echo "Installing Blender ${NAME}"
mkdir -p $CACHE
if [ ! -f $TAR ]; then
    wget -O $TAR $URL
fi
tar -xf $TAR -C $HOME

echo "export BLENDER_PATH=\"${HOME}/${NAME}\"" > .envs
echo "export PATH=${PATH}:${BLENDER_PATH}" >> .envs