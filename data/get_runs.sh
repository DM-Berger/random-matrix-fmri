find -iname "*bold.nii.gz" | grep -v afni | grep -v derivatives | grep -v prefrontal | grep -v sleep | grep -v slept | grep -v summed | grep -v interweave | grep -v -x 'echo-[23]' | sort -r 
