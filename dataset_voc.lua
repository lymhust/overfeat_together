require('LuaXML')
require 'image'
draw = require 'draw'
local imH, imW = 320, 320
local set = 'voc2007'
    
-- Functions
----------------------------------------------------------------------
function load_txt(fname)
    local name = {}
    local file = io.open(fname)
    local i = 0
    for v in file:lines() do
        i = i + 1
        name[i] = v 
    end
    file:close()
    return name
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

function load_data(name, folder)
    local totalnum = 1000--#name
    local reduceRatio = math.sqrt(0.3)
    local IMAGE = torch.Tensor(totalnum, 3, imH, imW):zero()
    local MASK = torch.Tensor(totalnum, imH/4, imW/4):fill(2)
    local BBOX = torch.Tensor(totalnum, 4, imH/4, imW/4):zero()

    for i = 1, totalnum do
      -- Load image
      local img = image.load(folder..'/JPEGImages/'..name[i]..'.jpg')
      print(i)
      local oriH, oriW = img:size(2), img:size(3)
      IMAGE[i] = image.scale(img, imW, imH)
      
      -- Load XML
      local xfile = xml.load(folder..'/Annotations/'..name[i]..'.xml')
      
      for _, node in pairs(xfile:find('annotation')) do
        if (node[node.TAG] == 'object') then
            local bbox = node:find('bndbox')
            local left = tonumber(bbox:find('xmin')[1])/oriW*imW
            local right = tonumber(bbox:find('xmax')[1])/oriW*imW
            local top = tonumber(bbox:find('ymin')[1])/oriH*imH
            local bottom = tonumber(bbox:find('ymax')[1])/oriH*imH
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
            -- Test
            --[[
            draw.drawBox(IMAGE[i], torch.round(top), torch.round(bottom), torch.round(left), torch.round(right), 1)
            image.save('./mask_pos.jpg', MASK[i][1])
            image.save('./mask_neg.jpg', MASK[i][2])
            image.save('./bbox.jpg', IMAGE[i])
            -]]
        end -- if
      end -- node    
    end -- totalnum
    return {IMAGE, MASK, BBOX}
end

----------------------------------------------------------------------
-- Load image name
local trainName = load_txt(set..'/train/ImageSets/Main/trainval.txt')
--local testName = load_txt('./test/ImageSets/Main/test.txt')
print('Train num: '..#trainName)
--print('Test num: '..#testName)

-- Load image data and gtruth
local trainData = load_data(trainName, set..'/train')
--local testData = load_data(testName, './test')

-- Normalization
local mean, std = normalize_global(trainData[1])
--normalize_global(testData[1], mean, std)

-- Save
torch.save('./detection_model/voc/'..set..'_trainData.t7', trainData)
--torch.save('./processed/'..set..'_testData.t7', testData)

-- Test
---[[
local ind = 4
local IMAGE = trainData[1]
local MASK = trainData[2]
local BBOX = trainData[3]
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
image.save('./overfeat_together/voc/test.jpg', img)
--]]




















