def kaist_results_file(boxes, filename):
    print('Writing KAIST result file')
    # filename = os.path.join(jobs_dir, '{:s}.txt'.format(result_name))
    with open(filename, 'wt') as f:
        for ii, bbs in enumerate(boxes):
            if bbs.size(0) == 0:
                continue
            for bb in bbs:                
                f.write('{:d},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n'.format(
                    ii+1, bb[0],bb[1],bb[2]-bb[0]+1,bb[3]-bb[1]+1,bb[4]))



def write_coco_format(boxes, filename):
	'''
		Format:
			[{
				"image_id" : int, "category_id" : int, "bbox" : [x,y,width,height], "score" : float,
			}]
	'''
	print('Write results in COCO format.')
	
	import json

	with open(filename, 'wt') as f:
		f.write( json.dumps(boxes, indent=4) )
