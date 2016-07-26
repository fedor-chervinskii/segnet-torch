--
--  Copyright (c) 2016, Fedor Chervinskii
--

require 'nn'
require 'models/MultiLogSoftMax'

local function add_block(cont,n_conv,sizes,wid,str,pad)
    local wid = wid or 3
    local str = str or 1
    local pad = pad or 1
    for i=1,n_conv do
        cont:add(nn.SpatialConvolution(sizes[i],sizes[i+1],wid,wid,str,str,pad,pad))
        cont:add(nn.SpatialBatchNormalization(sizes[i+1]))
        cont:add(nn.ReLU())
    end
    return cont
end

local function createModel(opt)

    conv_sizes = {3,64,64,128,128,256,256,256,512,512,512,512,512,512}

    encoder = nn.Sequential()
    pool = {}

    counter = 1
    for i=1,2 do
        sizes = {conv_sizes[counter],conv_sizes[counter+1],conv_sizes[counter+2]}
        encoder = add_block(encoder,2,sizes)
        counter = counter + 2
        pool[i] = nn.SpatialMaxPooling(2,2,2,2)
        encoder:add(pool[i])
    end
    for i=3,5 do
        sizes = {conv_sizes[counter],conv_sizes[counter+1],conv_sizes[counter+2],conv_sizes[counter+3]}
        encoder = add_block(encoder,3,sizes)
        counter = counter + 3
        pool[i] = nn.SpatialMaxPooling(2,2,2,2)
        encoder:add(pool[i])
    end

    decoder = nn.Sequential()

    counter = #conv_sizes
    for i=5,3,-1 do
        decoder:add(nn.SpatialMaxUnpooling(pool[i]))
        sizes = {conv_sizes[counter],conv_sizes[counter-1],conv_sizes[counter-2],conv_sizes[counter-3]}
        decoder = add_block(decoder,3,sizes)
        counter = counter - 3
    end
    for i=2,1,-1 do
        decoder:add(nn.SpatialMaxUnpooling(pool[i]))
        sizes = {conv_sizes[counter],conv_sizes[counter-1],conv_sizes[counter-2]}
        counter = counter - 2
        decoder = add_block(decoder,i,sizes)
    end
    decoder:add(nn.SpatialConvolution(conv_sizes[2],opt.nClasses,3,3,1,1,1))

    net = nn.Sequential()
    net:add(encoder)
    net:add(decoder)
    
    --print(tostring(net))
    --torch.save("Models/"..name, net)
    return net:cuda()
end

return createModel
