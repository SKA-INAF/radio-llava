import os
import json
import sys

# - Read args
filename_1= sys.argv[1]
filename_2= sys.argv[2]
outfile= "concat.json"

# - Open files
f1= open(filename_1, "r")
f2= open(filename_2, "r")

# - Read files
print("Reading file %s ..." % (filename_1))
datalist_1= json.load(f1)

print("Reading file %s ..." % (filename_2))
datalist_2= json.load(f2)

# - Creating a map of conversations with key "image" field
print("Creating a map of conversations from file %s ..." % (filename_2))
conv_map= {}
for item in datalist_2:
	image_path= item["image"] 
	conversations= item["conversations"]
	conv_map[image_path]= conversations
	
# - Concatenate conversations
pattern_to_be_removed= "<image>\n"

for idx, item in enumerate(datalist_1):
	image_path= item["image"]
	conversations= item["conversations"]
	
	if image_path in conv_map:
		conversations_to_be_added= conv_map[image_path]
		
		# - Remove <image>\n field
		for j, conversation in enumerate(conversations_to_be_added):
			value= conversation["value"]
			if pattern_to_be_removed in value:
				value_mod= value.replace(pattern_to_be_removed, "")
				conversations_to_be_added[j]["value"]= value_mod
		
		datalist_1[idx]["conversations"].extend(conversations_to_be_added)
	else:
		print("WARN: No match found for image %s, won't add conversations..." % (image_path))
		continue
		
# - Save update file
print("Convert and write updated datalist to file %s ..." % (outfile))
with open(outfile, "w") as fw: 
	json.dump(datalist_1, fw, indent=2)	
