# bug list

updated May 17, 2016

1.)	review all patches as there are outliers in there. Use the notebook checking patches visually.ipynb

2.) review what classes to be used

3.) review dataset with Shreyas
	. mismatch of samples between classes
	. absolute color values versus normalised
	. data augmentation
	.

4.) various patches are black. What to do with it.
    Solution could be to normalize all patches to xx.max() rather than float(255)