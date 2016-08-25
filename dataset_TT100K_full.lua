require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
json = require 'json'
draw = require 'draw'
local c = require 'trepl.colorize'
torch.setdefaulttensortype('torch.FloatTensor')

-- Functions
function normalize_submean(img, meanImg)
  return img - meanImg
end

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

function loaddata(ids, folder, content)
  print(c.red(folder..' folder number: '..#ids))
  
  -- Input image size
  local imH_l, imW_l = 2048, 2048
  local imH, imW = 480, 640
  local reduceRatio = math.sqrt(0.75)
  local batchSize = 2000
  local IMAGE = torch.Tensor(batchSize, 3, imH, imW):zero()
  local MASK_BOX = torch.Tensor(batchSize, 5, imH/4, imW/4):zero()
  
  -- Load img and gtruth
  local imgInfo = content.imgs
  local totalnum = #ids
  local ind = 1
  
  for t = 1, totalnum, batchSize do 
    local indPatch = 1
    
    for i = t, math.min(t+batchSize-1, totalnum) do
      -- Disp progress
      xlua.progress(i, totalnum)
      local imPath = imgInfo[ids[i]].path
      local img = image.load(imPath)
      img = image.scale(img, imW, imH)

      -- Image
      IMAGE[indPatch] = normalize_submean(img, meanImg)
      local obj = imgInfo[ids[i]].objects
      for j = 1, #obj do
        local bbox = obj[j].bbox
        local top = math.max(1, torch.round(bbox.ymin/imH_l*imH))
        local bottom = math.min(imH, torch.round(bbox.ymax/imH_l*imH))
        local left = math.max(1, torch.round(bbox.xmin/imW_l*imW))
        local right = math.min(imW, torch.round(bbox.xmax/imW_l*imW))
        --draw.drawBox(img, top, bottom, left, right, 1)
        local boxH = bottom-top+1
        local boxW = right-left+1
        local disH = torch.round((boxH-boxH*reduceRatio)/2)
        local disW = torch.round((boxW-boxW*reduceRatio)/2)
        local top_s = top+disH
        local bottom_s = bottom-disH
        local left_s = left+disW
        local right_s = right-disW
        -- Area ratio should around 0.75
        --print((bottom_s-top_s+1)*(right_s-left_s+1)/(boxH*boxW))
        top_s = torch.ceil(top_s/4)
        bottom_s = torch.ceil(bottom_s/4)
        left_s = torch.ceil(left_s/4)
        right_s = torch.ceil(right_s/4)
        --print(string.format('%f_%f_%f_%f',top,bottom,left,right))
        -- Mask
        MASK_BOX[{ indPatch,1,{top_s,bottom_s},{left_s,right_s} }] = 1
        -- BBox
        MASK_BOX[{ indPatch,2,{top_s,bottom_s},{left_s,right_s} }] = left/imW    -- Left
        MASK_BOX[{ indPatch,3,{top_s,bottom_s},{left_s,right_s} }] = right/imW   -- Right
        MASK_BOX[{ indPatch,4,{top_s,bottom_s},{left_s,right_s} }] = top/imH     -- Top
        MASK_BOX[{ indPatch,5,{top_s,bottom_s},{left_s,right_s} }] = bottom/imH  -- Bottom   
      end--obj
      --image.display(IMAGE[indPatch])
      --image.display(MASK_BOX[indPatch])
      indPatch = indPatch+1
    end--batchNum
    
    -- Save patch
    torch.save(string.format('./processed/%s_%d.t7',folder,ind), {IMAGE, MASK_BOX})
    ind = ind+1
    
  end--totalnum
end
-------------------------------------------------------------------------------------

-- Load meanImg
meanImg = torch.load('./processed/meanImg.t7')

-- Load json
local content = json.load('./annotations.json')
  
local set = {'train','test','other'}
for s = 1, #set do
  -- Load image ids
  local file = assert(io.open(set[s]..'/ids.txt', 'r'))
  local j = 0
  local ids = {}
  for i in file:lines() do
    table.insert(ids, i)
  end
  file:close()
  
  -- Load data
  loaddata(ids, set[s], content)
 
end--set




