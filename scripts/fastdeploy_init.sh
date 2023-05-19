# source this file to import libraries

PLATFORM=`uname`
FASTDEPLOY_LIBRARY_PATH=${BASH_SOURCE:-$0}
if [[ "$PLATFORM" = "Linux" ]];then
    FASTDEPLOY_LIBRARY_PATH=`readlink -f ${FASTDEPLOY_LIBRARY_PATH}`
fi
FASTDEPLOY_LIBRARY_PATH=$(cd `dirname ${FASTDEPLOY_LIBRARY_PATH}`; pwd)

echo "=============== Information ======================"
echo "FastDeploy Library Path: $FASTDEPLOY_LIBRARY_PATH"
echo "Platform: $PLATFORM"
echo "=================================================="

# Find all the .so files' path
if [[ "$(ps -a $$)" =~ "zsh" ]]; then
    ALL_SO_FILES=(`find $FASTDEPLOY_LIBRARY_PATH -name "*.so*"`)
    ALL_DYLIB_FILES=(`find $FASTDEPLOY_LIBRARY_PATH -name "*.dylib*"`)
else
    ALL_SO_FILES=`find $FASTDEPLOY_LIBRARY_PATH -name "*.so*"`
    ALL_DYLIB_FILES=`find $FASTDEPLOY_LIBRARY_PATH -name "*.dylib*"`
fi

for SO_FILE in $ALL_SO_FILES;do
    LIBS_DIRECTORIES+=(${SO_FILE%/*})
done

# Find all the .dylib files' path
# ALL_DYLIB_FILES=(`find $FASTDEPLOY_LIBRARY_PATH -name "*.dylib*"`)
for DYLIB_FILE in $ALL_DYLIB_FILES;do
    LIBS_DIRECTORIES+=(${DYLIB_FILE%/*})
done

# Remove the dumplicate directories
LIBS_DIRECTORIES=($(awk -v RS=' ' '!a[$1]++' <<< ${LIBS_DIRECTORIES[@]}))

# Print the dynamic library location and output the configuration file
IMPORT_PATH=""
output_file=${FASTDEPLOY_LIBRARY_PATH}/fastdeploy_libs.conf
rm -rf $output_file
for LIB_DIR in ${LIBS_DIRECTORIES[@]};do
    echo "Find Library Directory: $LIB_DIR"
    echo "$LIB_DIR" >> $output_file
    IMPORT_PATH=${LIB_DIR}":"$IMPORT_PATH
done

if [ -f "ascend_init.sh" ]
then
    source ascend_init.sh
fi

echo "[Execute] Will try to export all the library directories to environments, if not work, please try to export these path by your self."
PLATFORM=`uname`
if [[ "$PLATFORM" = "Linux" ]];then
  NEW_LIB_PATH=$(tr ":" "\n" <<< "${IMPORT_PATH}:$LD_LIBRARY_PATH" | sort | uniq | tr "\n" ":")
  export LD_LIBRARY_PATH=$NEW_LIB_PATH
fi
if [[ "$PLATFORM" = "Darwin" ]];then
  NEW_LIB_PATH=$(tr ":" "\n" <<< "${IMPORT_PATH}:$DYLD_LIBRARY_PATH" | sort | uniq | tr "\n" ":")
  export DYLD_LIBRARY_PATH=$NEW_LIB_PATH
fi
