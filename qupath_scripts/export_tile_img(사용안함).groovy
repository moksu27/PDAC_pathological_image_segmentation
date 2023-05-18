/**
 * Script to export image tiles (can be customized in various ways).
 */

// Get the current image (supports 'Run for project')
def imageData = getCurrentImageData()

// Define output path (here, relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def projectBaseDir = '/path/to/project2'
def pathOutput = buildFilePath(projectBaseDir, 'tiles', name)
mkdirs(pathOutput)

// Define output resolution in calibrated units (e.g. µm if available)
double requestedPixelSize = 0.5

// Convert output resolution to a downsample factor
double pixelSize = imageData.getServer().getPixelCalibration().getAveragedPixelSize()
double downsample = requestedPixelSize / pixelSize

// Create an exporter that requests corresponding tiles from the original & labelled image servers
new TileExporter(imageData)
    .downsample(downsample)   // Define export resolution
    .imageExtension('.tif')   // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(512)            // Define size of each tile, in pixels
    .annotatedTilesOnly(false) // If true, only export tiles if there is a (classified) annotation present
    .overlap(64)              // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)   // Write tiles to the specified directory

print 'Done!'