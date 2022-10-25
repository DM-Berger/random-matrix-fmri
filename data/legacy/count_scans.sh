find -iname "*bold.nii.gz" | grep -v afni | grep -v derivatives | grep -v prefrontal | grep -v sleep | grep -v slept | grep -v -x 'echo-[123]' | sort -r | wc -l
