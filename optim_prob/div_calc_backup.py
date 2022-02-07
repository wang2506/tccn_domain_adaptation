def div_roi(loc_model,bs,lr,l_dset=None,st=None,\
            device=device,dt=d_train): #divergence_run_one_iteration
    # # label modifications
    # tobj = LocalUpdate(device,bs=bs,lr=lr,epochs=1,\
    #         dataset=l_dset)
    # _,w,loss = tobj.train(net=loc_model.to(device))
    
s_w,s_loss = div_roi(deepcopy(start_net),bs=args.div_bs,\
        lr=args.div_lr,\
        l_dset=random.sample(sl_dset,args.div_bs))


class LocalUpdate(object):
    def __init__(self,device,bs,lr,epochs,dataset=None,indexes=None):
        self.device = device
        self.bs = bs
        self.lr = lr
        self.dataset = dataset
        self.indexes = indexes
        self.epochs = epochs
        # self.ldr_train = DataLoader(segmentdataset(dataset,indexes),batch_size=bs,shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()
        
    def train(self,net):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(),lr=self.lr)#, momentum=0.5,weight_decay=1e-4) #l2 penalty
        epoch_loss = []
        for epoch in range(self.epochs):
            batch_loss = []
            for batch_indx,(images,labels) in enumerate(self.ldr_train):
                images,labels = images.to(self.device),labels.to(self.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs,labels)
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net,net.state_dict(),(sum(batch_loss)/len(batch_loss))    
    