"""
Transformer model.
Author: Zhibo Zhu. Date: 07/20/2026.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from attention_block import multi_head_self_attn, multi_head_cross_attn
from refine_module import refine_module
from embedding import embedding
from positional_encoding import positional_encoding
from encoder_block import encoder_block
from decoder_block import decoder_block

class transformer(nn.Module):
    def __init__(self, input_channel, output_channel, bias, pos, d_model, omega, N, num_heads, W, H, activition=nn.ReLU, approach='sine', dropout_emb=0.0, dropout_enc=0.0, dropout_dec=0.0, dtype=torch.float32):
        """
        Class initialization.

        Args:
        input_channel (int):    Dimension of data points.
        output_channel (int):   Dimension of the embedded space.
        bias (bool):            Bias term flag.
        pos (ndarray):          Position vector (1D) or matrix (2D). If 1D: [0, H - 1]. If 2D: [0, W - 1] x [0, H - 1].
        d_model (int):          Embedding space dimension.
        omega (tensor):         Sampling mask tensor. [1, W, H].
        N (int):                Encoder/Decoder layers.
        num_heads (int):        Number of attention heads.
        W (int):                Full k-space width.
        H (int):                Full k-space height.
        activition (nn.Module): Activition function.
        approach (string):      Encoding approach.
        dropout_emb (float):    Embedding layer dropout rate.
        dropout_enc (float):    Encoding layer dropout rate.
        dropout_dec (float):    Decoding layer dropout rate.
        dtype (torch.dtype):    Parameter/compute dtype, e.g. torch.float32 (default), torch.bfloat16, torch.float16.
        """
        super().__init__()

        self.embedding = embedding(input_channel, output_channel, bias, activition, dropout_emb)
        PE = positional_encoding(pos, d_model, approach)
        # Undersample PE to match input data dimension.
        batch_size = omega.shape[0]
        omega = omega.view(batch_size, -1) # [batch_size, W * H]
        self.PE_enc = copy.deepcopy(PE)
        self.PE_enc.PE = torch.masked_select(PE.PE, omega.unsqueeze(-1).bool()).view(1, -1, d_model)
        # For low-res, slice the central 1/4 region: pos[:, y_start:y_end, x_start:x_end]
        # Keep only the middle quarter (remove 3/8 from each side).
        self.PE_LR = positional_encoding(pos[:, H * 3 // 8:-H * 3 // 8, W * 3 // 8:-W * 3 // 8], d_model, approach)
        # For high-res, we need the full mask.
        self.PE_HR = PE

        self.encoders = nn.ModuleList([encoder_block(d_model, num_heads, dropout_enc) for _ in range(N)])
        self.LR_decoders = nn.ModuleList([decoder_block(d_model, num_heads, W // 4, H // 4, dropout_dec) for _ in range(N)])
        self.HR_decoders = nn.ModuleList([decoder_block(d_model, num_heads, W, H, dropout_dec) for _ in range(N)])

        # Final output projection: d_model -> 2 (real + imaginary for k-space)
        self.predict = nn.Linear(d_model, 2)

        return

    def forward(self, ksp):
        """
        Transformer foward operation.

        Args:
        ksp (tensor):           Input k-space tensor, [batch_size, seq_length1, 2]
        omega (tensor):         Sampling mask, 1: sampled, 0: not sampled. [batch_size, seq_length2]
        """
        batch_size = ksp.size(0)

        # Linear embedding and PE ksp.
        input = self.embedding(ksp)
        input = self.PE_enc(input)

        # Pass into encoders.
        # TO-DO: Send undersampled data instead of zero-filled data. Therefore, omega should also be ignored during the encoder path.
        output_enc = input
        for ii in range(len(self.encoders)):
            # output_enc = self.encoders[ii](output_enc, omega)
            output_enc = self.encoders[ii](output_enc)
        # print(f"Encoder output shape: {output_enc.shape}")

        # Gather PE_p and output_enc for LR decoders.
        # Expand PE to match batch size [1, seq, d_model] -> [batch, seq, d_model]
        PE_LR = self.PE_LR.PE.expand(batch_size, -1, -1)
        output_LR_dec = output_enc
        for ii in range(len(self.LR_decoders)):
            output_LR_dec = self.LR_decoders[ii](PE_LR, output_LR_dec, LR=True)
        # print(f"LR decoder output shape: {output_LR_dec.shape}")

        # Gather PE_p and output_LR_dec for HR decoders.
        PE_HR = self.PE_HR.PE.expand(batch_size, -1, -1) # Expand to batch size
        # print(f"PE_HR shape {PE_HR.shape}")
        output_HR_dec = output_LR_dec # Need a look at the dimension of input to HR decoders.
        for ii in range(len(self.HR_decoders)):
            output_HR_dec = self.HR_decoders[ii](PE_HR, output_HR_dec, LR=False)

        # Final output projection to k-space [batch, seq, d_model] -> [batch, seq, 2]
        output_kspace = self.predict(output_HR_dec)

        return output_kspace

        # === ALTERNATIVE IMPLEMENTATIONS ===
        # To use these, comment out the return above and uncomment one section below

        # --- OPTION B: Cascading Query, Constant K/V from Encoder ---
        # Query=prev layer output, K/V=encoder output (constant), Self-Attn Q/K/V=from cross-attn
        # LR Decoders:
        #   query = self.PE_LR.PE
        #   for decoder in self.LR_decoders:
        #       query = decoder(query, output_enc, None, LR=True)
        #   output_LR_dec = query
        #
        # HR Decoders:
        #   query = self.PE.PE
        #   for decoder in self.HR_decoders:
        #       query = decoder(query, output_LR_dec, None, LR=False)
        #   output_HR_dec = query
        #
        # NOTE: Current decoder_block.forward() keeps K/V constant and only Q cascades

        # --- OPTION C: Full Cascade (All Q, K, V evolve through layers) ---
        # For this option, decoder_block would need modification to accept single input
        # that serves as both Q (for cross-attn) and K/V (fed from prev layer)
        # LR Decoders:
        #   x = self.PE_LR.PE
        #   for decoder in self.LR_decoders:
        #       x = decoder(x, x, None, LR=True)  # x as both Q and K/V
        #   output_LR_dec = x
        #
        # HR Decoders:
        #   x = self.PE.PE
        #   for decoder in self.HR_decoders:
        #       x = decoder(x, x, None, LR=False)  # x as both Q and K/V
        #   output_HR_dec = x
        #
        # NOTE: This requires both args to decoder to be the previous layer's output


def train_model(model, train_loader, optimizer, scheduler, device, criteria, log_dir="D:/models/ksp_transformer", epochs=300):
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir=log_dir)

    model.train()
    
    for epoch in range(epochs):
        running_total_loss = 0.0
        
        for batch_indx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            # Individual batch losses
            batch_MSE_loss = criteria(outputs, targets)
            
            # Total composite loss for backprop
            loss += batch_MSE_loss
            
            loss.backward()
            # Real-time logging of model health metrics to TensorBoard.
            if batch_indx % 50 == 0: # Log every 50 batches
                global_step = epoch * len(train_loader) + batch_indx
                log_model_health(model, writer, global_step)
            optimizer.step()
            
            # Accumulate values
            running_total_loss += loss.item()
        
        # Calculate final epoch averages
        num_batches = len(train_loader)
        avg_epoch_loss = running_total_loss / num_batches
        
        # Step scheduler based on average total loss
        scheduler.step(avg_epoch_loss)
        
        # Print averages for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        if (epoch + 1) % 20 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(), # Crucial for LR consistency
            }
            torch.save(checkpoint, 'models/checkpoint.pth')
            print(f"Epoch [{epoch+1}/{epochs}], Avg lss: {avg_epoch_loss:.6f}, LR: {current_lr:.6e}")

        # Initialize best_l1 if first epoch
        if epoch == 0:
            best_loss = avg_epoch_loss
            
        # Checkpoint based on average plain L1
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), 'models/best_intermediate_model.pth')
            print(f"--- New Best loss found: {best_loss:.6f} at Epoch {epoch} ---")

    print('Train completed.')

def log_model_health(model, writer, step):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # Log weight and bias values as histograms
            writer.add_histogram(f"Parameters/{name}", param.data, step)
            
            # Log gradient magnitudes as histograms (Crucial for spotting vanishing grads)
            writer.add_histogram(f"Gradients/{name}", param.grad, step)
            
            # Log Scalar Sparsity: percentage of elements near zero
            sparsity = (param.data.abs() < 1e-4).float().mean().item()
            writer.add_histogram(f"Sparsity/{name}", sparsity, step)