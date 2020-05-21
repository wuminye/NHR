# encoding: utf-8

import logging
import torch

from ignite.engine import Events, Engine
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage

from apex import amp

import time



def create_supervised_trainer(model, optimizer, loss_fn, use_cuda=True, loss_whole_img = False, swriter = None):

    if use_cuda:
        model.cuda()

    iters = 0


    def res_hook2(grad,iters = 0):
        iters = iters + 1
        swriter.add_scalars('gradient2', {'res_max':torch.max(grad).item(),
                        'res_min':torch.min(grad).item()}, iters)

    def res_hook(grad,iters = 0):
        iters = iters + 1
        swriter.add_scalars('gradient', {'res_max':torch.max(grad).item(),
                        'res_min':torch.min(grad).item()}, iters)
    
    def _update(engine, batch):

        #torch.cuda.synchronize()
        start_time = time.time()

        model.train()
        optimizer.zero_grad()
        

        in_points = batch[1].cuda()
        K = batch[2].cuda()
        T = batch[3].cuda()
        near_far_max_splatting_size = batch[5]
        num_points = batch[4]
        point_indexes = batch[0]
        target = batch[7].cuda()
        inds = batch[6]
        rgbs = batch[8].cuda()

        res,depth,features,dir_in_world,rgb,m_point_features = model(in_points, K, T,
                            near_far_max_splatting_size, num_points, rgbs, inds)

        res.register_hook(res_hook2)
        if not engine.state.rgb_project:
            m_point_features.register_hook(res_hook)

        

        depth_mask = depth[:,0:1,:,:].detach().clone()
        depth_mask = depth_mask + target[:,3:4,:,:]
        depth_mask[depth_mask>0] = 1
        depth_mask = torch.clamp(depth_mask,0,1.0)

        if not loss_whole_img:
            cur_mask = (target[:,4:5,:,:]*depth_mask).repeat(1,3,1,1)
        else:
            cur_mask = (target[:,4:5,:,:]).repeat(1,3,1,1)



        target[:,0:3,:,:] = target[:,0:3,:,:]*cur_mask


        
        res[:,0:3,:,:] = res[:,0:3,:,:]*cur_mask
        

        

        vis_rgb = res[0,0:3,:,:].detach().clone()
        vis_rgb[cur_mask[0]<0.1] += 0.5


        render_mask = res[0][3:4,:,:].detach().clone()
        render_mask[target[0,4:5,:,:]<0.1] += 0.5

        render_mask = torch.clamp(render_mask,0,1.0)

        
        res[:,3,:,:] = res[:,3,:,:] * target[:,4,:,:]

                
        
        
        loss1, loss2, loss3 = loss_fn(res, target[:,0:4,:,:])


        l = loss1 + loss2

        #l.backward()
        with amp.scale_loss(l, optimizer) as scaled_loss:
            scaled_loss.backward()


        optimizer.step()

        #torch.cuda.synchronize()
        end_time = time.time()



        iters = engine.state.iteration

        swriter.add_scalar('Loss/train_ploss',loss1.item(), iters)
        swriter.add_scalar('Loss/train_mseloss',loss2.item(), iters)
        swriter.add_scalar('Loss/train_gradloss',loss3.item(), iters)
        swriter.add_scalar('Loss/train_totalloss',l.item(), iters)
        swriter.add_scalar('Time/batch_time',end_time-start_time, iters)

        #swriter.add_scalars('gradient', {'p-features_max':torch.max(m_point_features.grad).item(),
        #                'p-features_min':torch.min(m_point_features.grad).item()}, iters)


        depth = (depth - torch.min(depth))
        depth = depth / torch.max(depth)



        swriter.add_image('vis/rendered', res[0,0:3,:,:], iters)
        swriter.add_image('vis/GT', target[0][0:3,:,:], iters)
        swriter.add_image('vis/depth', depth[0], iters)
        swriter.add_image('vis/GT_Mask', target[0][3:4,:,:], iters)
        swriter.add_image('vis/rendered_mask', render_mask, iters)

        if engine.state.rgb_project:
            swriter.add_image('vis/RGB_PCPR', features[0], iters)




        return l.item()

    return Engine(_update)

def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        swriter
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    output_dir = cfg.OUTPUT_DIR
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("rendering_model.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, loss_whole_img = cfg.SOLVER.LOSS_WHOLE_IMAGE,swriter = swriter )


   
    checkpointer = ModelCheckpoint(output_dir, 'nr', checkpoint_period, n_saved=15, require_empty=False)

    checkpointer_tmp = ModelCheckpoint(output_dir, 'nr_tmp', 3000, n_saved=2, require_empty=False)

    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,'optimizer': optimizer})
    trainer.add_event_handler(Events.ITERATION_COMPLETED, checkpointer_tmp, {'model': model,'optimizer': optimizer})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')




    @trainer.on(Events.STARTED)
    def start_training(engine):
        if cfg.MODEL.RESUME_EPOCH > 0:
            engine.state.epoch = cfg.MODEL.RESUME_EPOCH 
            checkpointer._iteration  = cfg.MODEL.RESUME_EPOCH 
        engine.state.rgb_project = cfg.MODEL.NO_FEATURE_ONLY_RGB
        engine.state.pbeg = time.time()


    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        global pbeg
        if iter % log_period == 0:

            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.2f} Base Lr: {:.2e} time: {:.5f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], scheduler.get_lr()[0], time.time()-engine.state.pbeg))
            engine.state.pbeg = time.time()
    

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()


                


    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
