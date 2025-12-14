#pragma once

// Hàm tiện ích để tính chỉ số trong mảng phẳng từ chỉ số đa chiều
inline int get_idx(int n, int c, int h, int w, int C, int H, int W)
{
	return n * (C * H * W) + c * (H * W) + h * W + w;
}