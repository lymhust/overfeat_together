require 'image'
require 'imgraph'
local myutils = {}

function myutils.drawBox(img, top, bottom, left, right, lineWidth, color)
  if color == nil then
    color = {0, 1, 0}
  end
  local imsize = #img
  local imC, imW, imH = imsize[1], imsize[3], imsize[2]
  local topleft, bottomright = {}, {} 
  topleft.x = top
  topleft.y = left
  bottomright.x = bottom
  bottomright.y = right
  if imC == 3 then
    -- Line one
    img[1][{ {topleft.x, bottomright.x},{topleft.y, topleft.y+lineWidth} }] = color[1]
    img[2][{ {topleft.x, bottomright.x},{topleft.y, topleft.y+lineWidth} }] = color[2]
    img[3][{ {topleft.x, bottomright.x},{topleft.y, topleft.y+lineWidth} }] = color[3]
    -- Line two
    img[1][{ {topleft.x, bottomright.x},{bottomright.y-lineWidth, bottomright.y} }] = color[1]
    img[2][{ {topleft.x, bottomright.x},{bottomright.y-lineWidth, bottomright.y} }] = color[2]
    img[3][{ {topleft.x, bottomright.x},{bottomright.y-lineWidth, bottomright.y} }] = color[3]
    -- Line three
    img[1][{ {topleft.x, topleft.x+lineWidth},{topleft.y, bottomright.y} }] = color[1]
    img[2][{ {topleft.x, topleft.x+lineWidth},{topleft.y, bottomright.y} }] = color[2]
    img[3][{ {topleft.x, topleft.x+lineWidth},{topleft.y, bottomright.y} }] = color[3]
    -- Line four
    img[1][{ {bottomright.x-lineWidth, bottomright.x},{topleft.y, bottomright.y} }] = color[1]
    img[2][{ {bottomright.x-lineWidth, bottomright.x},{topleft.y, bottomright.y} }] = color[2]
    img[3][{ {bottomright.x-lineWidth, bottomright.x},{topleft.y, bottomright.y} }] = color[3]
  else
    local colorgray
    if type(color) == 'number' then
      colorgray = color
    else
      colorgray = 0.2989*color[1] + 0.5870*color[2] + 0.1140*color[3]
    end
    -- Line one
    img[1][{ {topleft.x, bottomright.x},{topleft.y, topleft.y+lineWidth} }] = colorgray
    -- Line two
    img[1][{ {topleft.x, bottomright.x},{bottomright.y-lineWidth, bottomright.y} }] = colorgray
    -- Line three
    img[1][{ {topleft.x, topleft.x+lineWidth},{topleft.y, bottomright.y} }] = colorgray
    -- Line four
    img[1][{ {bottomright.x-lineWidth, bottomright.x},{topleft.y, bottomright.y} }] = colorgray
  end

  return img
end

function myutils.bwlabel(img)
    local graph = imgraph.graph(img:float())
    graph = imgraph.connectcomponents(graph, 0.1) + 10
    graph = torch.cmul(graph, img:float())
    local location = graph:nonzero()
    local val = graph[location[1][1]][location[1][2]]
    local indx = {}
    table.insert(indx, val)
    for i = 1, location:size(1) do
        local tmp = graph[location[i][1]][location[i][2]]
        local flag = true
        for j = 1, #indx do
            if (tmp == indx[j]) then
                flag = false
                break
            end
        end
        if (flag == true) then
	        val = tmp
	        table.insert(indx, val)
        end
    end
    return graph, indx
end

return myutils























