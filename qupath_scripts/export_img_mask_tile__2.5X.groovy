import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def projectBaseDir = '/Users/kim/Desktop/pathology_project/bsm/PDA_mask_img'
def pathOutput = buildFilePath(projectBaseDir, '2.5X', name)
mkdirs(pathOutput)

// Define output resolution
double requestedPixelSize = 4

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()


// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.BLACK) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Tumor', 1)      // Choose output labels (the order matters!)
    .build()


// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.png')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(512)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(64)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory
    
// print downsample    
// print imageData.getServer().getMetadata().getWidth()/downsample
// print imageData.getServer().getMetadata().getHeight()/downsample
print 'Done!'