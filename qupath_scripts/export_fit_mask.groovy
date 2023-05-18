def imageData = getCurrentImageData()

// Define output path (relative to project)
def projectBaseDir = '/Users/kim/Desktop/pathology_project/bsm/PDA_mask_img'
def outputDir = buildFilePath(projectBaseDir, 'fit')
mkdirs(outputDir)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def path = buildFilePath(outputDir+'/'+name, name + "-labelled.png")

// Define how much to downsample during export (may be required for large images)
double downsample = 8

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
  .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
  .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
  .addLabel('Tumor', 1)      // Choose output labels (the order matters!)
  .build()

// Write the image
writeImage(labelServer, path)
print 'Done!'