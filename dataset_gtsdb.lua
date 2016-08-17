require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
draw = require 'draw'
local c = require 'trepl.colorize'
torch.setdefaulttensortype('torch.FloatTensor')

GTFILE = './resized/gt.txt'
IMFILE = './resized/'

-- Functions
function normalize_global(dataset, mean, std)
  local std = std or dataset:std()
  local mean = mean or dataset:mean()
  dataset:add(-mean)
  dataset:div(std)
  return mean, std
end

function normalize_local(dataset)
  local norm_kernel = image.gaussian1D(7)
  local norm = nn.SpatialContrastiveNormalization(3,norm_kernel)
  local batch = 10 -- Can be reduced if you experience memory issues
  local dataset_size = dataset:size(1)
  for i=1,dataset_size,batch do
    local local_batch = math.min(dataset_size,i+batch) - i
    local normalized_images = norm:forward(dataset:narrow(1,i,local_batch))
    dataset:narrow(1,i,local_batch):copy(normalized_images)
  end
end
-------------------------------------------------------------------------------------

-- Input image size
local imH, imW = 480, 640

-- Create ground truth
GTRUTH = {}
local f = assert(io.open(GTFILE, 'r'))
local imnameold = '00000.ppm'
local tmpgtruth = {}

while(1) do
  local line = f:read("*line")
  if (line == nil) then
    break
  end 
  
  -- Get ground truth
  local gtruth = string.split(line, ';')
  local imname = gtruth[1]
  local leftcol = tonumber(gtruth[2])
  local toprow = tonumber(gtruth[3])
  local rightcol = tonumber(gtruth[4])
  local bottomrow = tonumber(gtruth[5])
  
  -- Get image size
  local imrow = 800
  local imcol = 1360
  local indleft = torch.round(leftcol / imcol * imW)
  local indright = torch.round(rightcol / imcol * imW)
  local indtop = torch.round(toprow / imrow * imH)
  local indbottom = torch.round(bottomrow / imrow * imH)
  
  if (imname == imnameold) then
		table.insert(tmpgtruth, {indleft, indright, indtop, indbottom})
  else
    table.insert(GTRUTH, {imnameold, tmpgtruth})
    tmpgtruth = {}
  	imnameold = imname
  	table.insert(tmpgtruth, {indleft, indright, indtop, indbottom})
  end
end
print(GTRUTH)

-- Create labels for full image
local totalnum = #GTRUTH
local reduceRatio = math.sqrt(0.5)
local IMAGE = torch.Tensor(totalnum, 3, imH, imW):zero()
local MASK = torch.Tensor(totalnum, imH/4, imW/4):fill(2)
local BBOX = torch.Tensor(totalnum, 4, imH/4, imW/4):zero()

local indpatch = 1

for i = 1, totalnum do
  
  local tmpgruth = GTRUTH[i]
  local imname = tmpgruth[1]
  local img = image.load(IMFILE..imname)
  -- Image
  IMAGE[i] = img
  local left, right, top, bottom
  -- For one image
  for k = 1, #tmpgruth[2] do
    local tmptable = tmpgruth[2][k]
    left = tmptable[1]
    right = tmptable[2]
    top = tmptable[3]
    bottom = tmptable[4]
    local boxH = bottom-top+1
    local boxW = right-left+1
    local disH = torch.round((boxH-boxH*reduceRatio)/2)
    local disW = torch.round((boxW-boxW*reduceRatio)/2)
    local top_s = top+disH
    local bottom_s = bottom-disH
    local left_s = left+disW
    local right_s = right-disW
    -- Area ratio should around 0.5
    --print((bottom_s-top_s+1)*(right_s-left_s+1)/(boxH*boxW))
    top_s = torch.ceil(top_s/4)
    bottom_s = torch.ceil(bottom_s/4)
    left_s = torch.ceil(left_s/4)
    right_s = torch.ceil(right_s/4)
    -- Mask
    MASK[{ i,{top_s,bottom_s},{left_s,right_s} }] = 1
    -- BBox
    BBOX[{ i,1,{top_s,bottom_s},{left_s,right_s} }] = left/imW    -- Left
    BBOX[{ i,2,{top_s,bottom_s},{left_s,right_s} }] = right/imW   -- Right
    BBOX[{ i,3,{top_s,bottom_s},{left_s,right_s} }] = top/imH     -- Top
    BBOX[{ i,4,{top_s,bottom_s},{left_s,right_s} }] = bottom/imH  -- Bottom
  end
	print(i)
end

-- Normalize
print(c.green'Normalizing')
local mean, std = normalize_global(IMAGE)
--torch.save('./data/MEAN_STD.t7', {mean, std})
--normalize_local(IMAGE)

-- Save data
print(c.green'Saving')
local data = {IMAGE, MASK, BBOX}
torch.save('./gtsdb.t7', data)
print(c.green'Finished')

-- Test data bbox
---[[
local ind = 2
local img = IMAGE[ind]
local mask = MASK[ind]
for i = 1, mask:size(1) do
	for j = 1, mask:size(2) do
		if (mask[i][j] == 1) then
			local left = torch.round(BBOX[{ ind,1,i,j }]*imW)
			local right = torch.round(BBOX[{ ind,2,i,j }]*imW)
			local top = torch.round(BBOX[{ ind,3,i,j }]*imH)
			local bottom = torch.round(BBOX[{ ind,4,i,j }]*imH)
			draw.drawBox(img, top, bottom, left, right, 1, {1,0,0})
		end
	end
end
mask = image.scale(mask, imW, imH, 'simple')
for i = 1, 3 do
  local tmp = img[i]
  tmp[mask:eq(1)] = 1
  img[i] = tmp
end
image.save('./test.jpg', img)
--]]
