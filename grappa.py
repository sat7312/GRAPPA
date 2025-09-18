import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.linalg import lstsq
from numpy.fft import ifft2

# 1. MATLAB 데이터
# DATA : (200, 200, 8)
def load_kspace_mat(filepath):
    data = sio.loadmat(filepath)
    kspace = data['DATA']
    kspace = np.asarray(kspace)
    return kspace

# 2. ACS 영역 추출
# nx = phase-encoding(행), ny = frequency-encoding(열)
# cx, cy : k-space의 중심 좌표
# ax, ay : ACS의 중심에서 절반 (중앙을 기준으로 좌/우, 상/하로 얼마나 자를지)
# (cx, cy)를 기준으로 좌우 ax, 상하 ay만큼 slice
def extract_acs(kspace, acs_size = (30, 30)):
    nx, ny, _ = kspace.shape
    cx, cy = nx // 2, ny // 2
    ax, ay = acs_size[0] // 2, acs_size[1] // 2
    return kspace[cx - ax:cx + ax, cy - ay:cy + ay, :]

# 3. 가중치 계산
# kx, ky : 커널의 세로, 가로
# sx, sy : ACS의 높이, 너비
# num_x, num_y : sliding 가능한 x, y방향 수
def grappa_calibrate(acs, kernel=(4, 2), accel = 2):
    kx, ky = kernel
    sx, sy, _ = acs.shape
    num_x = sx - kx + 1
    num_y = sy - accel * (ky - 1)
    input = []  # 입력 데이터
    target_center = []  # 출력 타겟 (보간하려는 위치)

    for x in range(kx//2,sx-kx//2):
        for y in range(accel * (ky//2),sy-accel*(ky//2)):
            # 커널 크기만큼 더한 범위 -> 커널 사이즈만큼 적용 (sliding window마다 input에 대입)
            window = acs[x - kx // 2 : x + kx // 2, 
                         y - accel * (ky // 2) + 1 : y + accel * (ky // 2) - 1 : accel, :]
            # print(window.shape)
            center = acs[x , y , :]
            input.append(window.flatten())  #flatten 안하면 차원 불일치하므로 꼭 하기
            target_center.append(center)
    input = np.array(input)
    target_center = np.array(target_center)
    # print(input.shape)
    # print(target_center.shape)

    # (x, residuals, rank, s) = lstsq(A, B)
    # A @ weights ≈ B를 만족하는 weights를 least squares로 계산 (최소제곱 해 계산)
    weights, _, _, _ = lstsq(input, target_center)
    # print(weights)
    return weights, kernel

# 4. GRAPPA 적용
def grappa_reconstruct(kspace_us, weights, kernel, accel = 2):
    sx, sy, _ = kspace_us.shape
    kx, ky = kernel
    output = np.copy(kspace_us)

    # print(sx, sy)
    # print(kx, ky)

    # patch : 현재 보간 대상 위치 주변의 sampling data
    for x in range(kx // 2, sx - kx // 2):
        for y in range(ky // 2 * accel, sy - ky // 2 * accel,  accel):
            if np.all(output[x, y, :] == 0):  # 복원 위치(x, y)가 0이면 보간
                patch = output[x - kx // 2 : x + kx // 2,
                               y - accel * (ky // 2) + 1 : y + accel * (ky // 2) - 1: accel, :]
                # print(patch.flatten())
                output[x, y, :] = patch.flatten()[np.newaxis,:] @ weights # 선형 곱 (여기도 flatten 안하면 차원 불일치)
                # print(patch.flatten()[np.newaxis,:] @ weights)
    # print((patch.flatten() @ weights).shape)
    # print(patch.flatten().shape)
    # print(weights.shape)
    return output

# 5. 이미지 복원
# axis = 0 -> phase encoding(행)
# axis = 1 -> frequency encoding(열)
# axis = 2 -> coil
def reconstruct_image(kspace_full):
    img_coil = np.fft.fftshift(ifft2(np.fft.fftshift(kspace_full), axes = (0, 1)))
    img_final = np.sqrt(np.sum(np.abs(img_coil) ** 2, axis = -1))   #RSS 구하기
    return img_final

if __name__ == "__main__":
    filepath = "brain_8ch.mat"
    kspace = load_kspace_mat(filepath)

    # 가속률로 undersampling (R = 2)
    kspace_undersampled = np.copy(kspace)
    kspace_undersampled[:, ::2, :] = 0

    # plt.imshow(np.log(1 + np.abs(kspace_undersampled[:, :, 0])), cmap='gray')
    # plt.savefig("grappa_us.png")

    # ACS 추출 -> weight 계산 -> GRAPPA -> 이미지 재구성
    acs = extract_acs(kspace, acs_size = (30, 30))
    weights, kernel = grappa_calibrate(acs, kernel=(4, 4), accel = 2)

    kspace_recon = grappa_reconstruct(kspace_undersampled, weights, kernel, accel = 2)
    image = reconstruct_image(kspace_recon)

    # plt.imshow(np.log(1 + np.abs(kspace_recon[:, :, 0])), cmap='gray')
    # plt.savefig("grappa_recon_k.png")

    plt.imshow(np.abs(image), cmap='gray')
    plt.savefig("grappa_output.png")