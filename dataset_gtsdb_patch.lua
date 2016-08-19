require 'torch'
require 'image'
require 'gnuplot'
require 'nn'
draw = require 'draw'
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
  local indleft = torch.round(leftcol / imcol * 640)
  local indright = torch.round(rightcol / imcol * 640)
  local indtop = torch.round(toprow / imrow * 480)
  local indbottom = torch.round(bottomrow / imrow * 480)
  
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

-- Create image patches
local patchsize = 128
local step = 64
local masksize = patchsize / 4
local totalnum = 3255
local overlap = 0.75
IMAGEPATCH = torch.Tensor(totalnum, 3, patchsize, patchsize):zero()
IMAGEMASK = torch.Tensor(totalnum, 5, masksize, masksize):zero()
PATCHSCORE = torch.Tensor(totalnum, 1):zero()
local indpatch = 1

for i = 1, #GTRUTH do

    local tmpgruth = GTRUTH[i]
    local imname = tmpgruth[1]
    local img = image.load(IMFILE..imname)
    local imgmask = torch.Tensor(480, 640):zero()
    local area = {}
    local indleft, indright, indtop, indbottom
  
    for k = 1, #tmpgruth[2] do
        tmptable = tmpgruth[2][k]
        indleft = tmptable[1]
        indright = tmptable[2]
        indtop = tmptable[3]
        indbottom = tmptable[4]
        local tmparea = (indbottom-indtop+1) * (indright-indleft+1)
        table.insert(area, tmparea)
        imgmask[{ {indtop,indbottom},{indleft,indright} }] = k
    end
  
    for indr = 1, 480, step do
  	    if (indr+patchsize-1) > 480 then indr = 480-patchsize+1 end
  	
  	for indc = 1, 640, step do
  	    if (indc+patchsize-1) > 640 then indc = 640-patchsize+1 end
  	  
  	        -- Load image mask 		
      		local tmpmask = imgmask[{ {indr,indr+patchsize-1},{indc,indc+patchsize-1} }]
      		local tmpid = tmpmask:max()
      		local score
      		
      		if tmpid == 0 then
      		    score = 0
      		else
      		    for idx = 1, tmpid do
      		      local tmpscore = torch.Tensor(tmpid):zero()
      		        tmpscore[idx] = tmpmask:eq(idx):sum()/area[idx]
      				score = tmpscore:max()
      			end
      		end
  		
  		    -- Select data
  		    if (score >= overlap) then
			    IMAGEPATCH[{ indpatch,{},{},{} }] = img[{ {},{indr,indr+patchsize-1},{indc,indc+patchsize-1} }] 
			    local mask = image.scale(tmpmask, masksize, masksize,'simple')
			    local tmp = torch.Tensor(mask:size()):zero()
			    tmp[mask:gt(0)] = 1
			    IMAGEMASK[{ indpatch,1,{},{} }] = tmp
				
				for idx = 1, tmpmask:max() do			
					local location_large = tmpmask:eq(idx):nonzero()
					local location_small = mask:eq(idx):nonzero()
					if (location_small:dim() > 0) then
						local top = location_large[1][1] / patchsize
						local bottom = location_large[-1][1] / patchsize
						local left = location_large[1][2] / patchsize
						local right = location_large[-1][2]	/ patchsize
						
						local top_s = location_small[1][1]
						local bottom_s = location_small[-1][1]
						local left_s = location_small[1][2]
						local right_s = location_small[-1][2]
						
            -- Generate bbox
--[[
						IMAGEMASK[{ indpatch,2,{top_s,bottom_s},{left_s,right_s} }] = top
						IMAGEMASK[{ indpatch,3,{top_s,bottom_s},{left_s,right_s} }] = bottom
						IMAGEMASK[{ indpatch,4,{top_s,bottom_s},{left_s,right_s} }] = left
						IMAGEMASK[{ indpatch,5,{top_s,bottom_s},{left_s,right_s} }] = right					
--]]	
  					-- Generate patial mask
						local r_half = torch.floor(top_s+(bottom_s-top_s)/2)
						local c_half = torch.floor(left_s+(right_s-left_s)/2) 
						IMAGEMASK[{ indpatch,2,{top_s,r_half},{left_s,right_s} }] = 1--top
						IMAGEMASK[{ indpatch,3,{r_half,bottom_s},{left_s,right_s} }] = 1--bottom
						IMAGEMASK[{ indpatch,4,{top_s,bottom_s},{left_s,c_half} }] = 1--left
						IMAGEMASK[{ indpatch,5,{top_s,bottom_s},{c_half,right_s} }] = 1--right

					end
				end
				
				PATCHSCORE[indpatch] = score
				indpatch = indpatch + 1
				print(indpatch)
  		end
  		
        if (indc+patchsize-1) == 640 then break end
  	end
  	
    if (indr+patchsize-1) == 480 then break end
  end
	
end

-- Normalize
local mean, std = normalize_global(IMAGEPATCH)
--torch.save('./MEAN_STD.t7', {mean, std})
normalize_local(IMAGEPATCH)

local pos = (1 - PATCHSCORE:eq(0):sum() / totalnum) * 100
print('Positive pencentage: '..pos..'%')

-- Save data
local data = {IMAGEPATCH, IMAGEMASK}
torch.save('./gtsdb_patch.t7', data)

-- Test data bbox
--[[
local ind = 1
local img = IMAGEPATCH[{ ind,{},{},{} }]
local mask = IMAGEMASK[{ ind,{1},{},{} }]
for i = 1, mask:size(2) do
	for j = 1, mask:size(3) do
		if (mask[{ 1,i,j }] >= 0.95) then
			local top = IMAGEMASK[{ ind,2,i,j }]
			local bottom = IMAGEMASK[{ ind,3,i,j }]
			local left = IMAGEMASK[{ ind,4,i,j }]
			local right = IMAGEMASK[{ ind,5,i,j }]
			draw.drawBox(img, top, bottom, left, right, 1)
		end
	end
end
image.display(img)
image.display(mask)
print(PATCHSCORE[ind])
--]]

-- Test data patial mask
---[[
local ind = 1
local img = IMAGEPATCH[{ ind,{},{},{} }]
image.display(img)
image.display(IMAGEMASK[{ ind,{},{},{} }])
--]]
