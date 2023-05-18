def imageData = getCurrentImageData()
// Write the full image downsampled by a factor of 8
def server = getCurrentServer()
def requestFull = RegionRequest.createInstance(server, 8)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def projectBaseDir = '/Users/kim/Desktop/pathology_project/bsm/PDA_mask_img'
def pathOutput = buildFilePath(projectBaseDir, 'fit', name)
mkdirs(pathOutput)
writeImageRegion(server, requestFull, projectBaseDir+'/fit/'+name+'/'+name+'.png')
print 'Done!'