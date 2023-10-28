set -e

NAME="blender-3.6.5-linux-x64"
NAMETAR="${NAME}.tar.xz"
CACHE="${HOME}/.blender-cache"
TAR="${CACHE}/${NAMETAR}"
URL="https://mirror.clarkson.edu/blender/release/Blender3.6/${NAMETAR}"

echo "Installing Blender ${NAME}"
mkdir -p $CACHE
if [ ! -f $TAR ]; then
    wget -O $TAR $URL -q
fi
tar -xf $TAR -C $HOME

echo "export PATH=${PATH}:\"${HOME}/${NAME}\"" > .envs