#include "include/d3dApp.h"
#include "include/MathHelper.h"
#include "include/UploadBuffer.h"

struct Vertex {
	DirectX::XMFLOAT3 position;
	DirectX::XMFLOAT4 color;
};

struct ObjectConstants {
    DirectX::XMFLOAT4X4 WVP = MathHelper::Identity4x4();
};

class RainbowBoxDemo : public D3DApp {
public:
    RainbowBoxDemo(HINSTANCE hInstance);
    RainbowBoxDemo(const RainbowBoxDemo& rValue) = delete;
    RainbowBoxDemo& operator=(const RainbowBoxDemo& rValue) = delete;
    ~RainbowBoxDemo();

    virtual bool Initialize() override;

private:
    virtual void OnResize() override;
    virtual void Update(const GameTimer& timer) override;
    virtual void Draw(const GameTimer& timer) override;

    virtual void OnMouseDown(WPARAM buttonState, int x, int y) override;
    virtual void OnMouseUp(WPARAM buttonState, int x, int y) override;
    virtual void OnMouseMove(WPARAM buttonState, int x, int y) override;
    void UpdateCamera(const GameTimer& timer);

    void BuildDescriptorHeaps();
    void BuildConstantBuffers();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildBoxGeometry();
    void BuildPSO();

private:
    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mCbvHeap = nullptr;
    std::unique_ptr<UploadBuffer<ObjectConstants>> mObjectCB = nullptr;
    std::unique_ptr<MeshGeometry> mBoxGeo = nullptr;
    Microsoft::WRL::ComPtr<ID3DBlob> mvsByteCode = nullptr;
    Microsoft::WRL::ComPtr<ID3DBlob> mpsByteCode = nullptr;
    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;
    Microsoft::WRL::ComPtr<ID3D12PipelineState> mPSO = nullptr;

    DirectX::XMFLOAT4X4 mWorld = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 mView = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 mProjection = MathHelper::Identity4x4();

    // Actually Pitch and Yaw aren't perfect terms here, but close enough.
    float mCameraPitch = 1.5f * DirectX::XM_PI;
    float mCameraYaw = DirectX::XM_PIDIV4;
    float mCameraDistFromCube = 5.f;

    POINT mLastMousePos;
};

/* Uncomment to run this file as the main. Only one WinMain can exist in the project.
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
// Runtime memory check for debug builds
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try {
        RainbowBoxDemo app(hInstance);
        if (!app.Initialize()) return 0;
        return app.Run();
    }
    catch (DxException& e) {
        MessageBox(nullptr, e.ToString().c_str(), L"Caught failed HResult!", MB_OK);
        return 0;
    }
}
//*/


RainbowBoxDemo::RainbowBoxDemo(HINSTANCE hInstance) : D3DApp(hInstance) {}

RainbowBoxDemo::~RainbowBoxDemo() {}

bool RainbowBoxDemo::Initialize() {
    if (!D3DApp::Initialize()) return false;

    // Start with a fresh cmd list
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    // todo??: does order matter here?
    BuildDescriptorHeaps();
    BuildConstantBuffers();
    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildBoxGeometry(); // this call adds to the command list
    BuildPSO();

    // Execute the commands added by BuildBoxGeometry
    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // wait for init to finish
    FlushCommandQueue();

    return true;
}

void RainbowBoxDemo::OnResize() {
    D3DApp::OnResize();

    // Need to recompute aspect ratio and projection matrix for new window size
    DirectX::XMMATRIX P = DirectX::XMMatrixPerspectiveFovLH(
        0.25f * MathHelper::Pi, AspectRatio(), 1.f, 1000.f);
    XMStoreFloat4x4(&mProjection, P);
}

void RainbowBoxDemo::Update(const GameTimer& timer) {
    UpdateCamera(timer); // This updates mView 

    DirectX::XMMATRIX W = XMLoadFloat4x4(&mWorld);
    DirectX::XMMATRIX P = XMLoadFloat4x4(&mProjection);
    DirectX::XMMATRIX V = XMLoadFloat4x4(&mView);
    DirectX::XMMATRIX WVP = W * V * P;

    // Push updated matrices (world, view, proj) to the constant buffer
    ObjectConstants objConstants;
    XMStoreFloat4x4(&objConstants.WVP, XMMatrixTranspose(WVP));
    mObjectCB->CopyData(0, objConstants);
}

void RainbowBoxDemo::Draw(const GameTimer& timer) {
    // The command queue is flushed at the end of each Draw call so this is safe.
    // Although that's inefficient so apparently we find a better way to do it later on.
    ThrowIfFailed(mDirectCmdListAlloc->Reset());

    // Re-use command list memory; allocator was already reset so this is safe
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), mPSO.Get()));

    // Our first command is to transition the current back buffer into "render target" state
    // Currently it's in a "present" state leftover from being front buffer (last frame 
    // changed it to back buffer from a Present() command)
    CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
        CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET
    );
    mCommandList->ResourceBarrier(1, &transition);

    // Gotta set the viewport and scissor rects again since we reset the command list
    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    // Clear the back and depth/stencil buffers
    mCommandList->ClearRenderTargetView(CurrentBackBufferView(),
        DirectX::Colors::Bisque, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(),
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.f, 0, 0, nullptr);

    // Set the pipeline: bind the back buffer as render target and use main DSV buffer 
    D3D12_CPU_DESCRIPTOR_HANDLE hBackBufferView = CurrentBackBufferView();
    D3D12_CPU_DESCRIPTOR_HANDLE hDepthStencilView = DepthStencilView();
    mCommandList->OMSetRenderTargets(1, &hBackBufferView, true, &hDepthStencilView);

    ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

    mCommandList->IASetVertexBuffers(0, 1, &mBoxGeo->VertexBufferView());
    mCommandList->IASetIndexBuffer(&mBoxGeo->IndexBufferView());
    mCommandList->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);

    mCommandList->SetGraphicsRootDescriptorTable(0, mCbvHeap->GetGPUDescriptorHandleForHeapStart());

    mCommandList->DrawIndexedInstanced(
        mBoxGeo->DrawArgs["box"].IndexCount,
        1, 0, 0, 0);

    // Transition the back buffer back to "present" state in preperation to swap 
    transition = CD3DX12_RESOURCE_BARRIER::Transition(
        CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_RENDER_TARGET,
        D3D12_RESOURCE_STATE_PRESENT
    );
    mCommandList->ResourceBarrier(1, &transition);

    ThrowIfFailed(mCommandList->Close());  // END OF CMD LIST

    // Add command list to command queue and submit to GPU for execution 
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);
    // Remember at this point the cmds are just submitted to the queue, NOT already executed

    // Now swap the buffers **(why do this before we know the cmd queue is flushed??)**
    ThrowIfFailed(mSwapChain->Present(0, 0));
    mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

    // Flush so next Draw command can re-use allocator memory.
    // Inefficient, we will learn a better way to do this
    FlushCommandQueue();
}

void RainbowBoxDemo::OnMouseDown(WPARAM buttonState, int x, int y) {
    mLastMousePos.x = x;
    mLastMousePos.y = y;

    SetCapture(mhMainWnd);
}

void RainbowBoxDemo::OnMouseUp(WPARAM buttonState, int x, int y) {
    ReleaseCapture();
}

void RainbowBoxDemo::OnMouseMove(WPARAM buttonState, int x, int y) {
    if ((buttonState & MK_LBUTTON) != 0) {
        // Left click-and-drag = ROTATE.
        float xDiff = DirectX::XMConvertToRadians(0.25f * (x - mLastMousePos.x));
        float yDiff = DirectX::XMConvertToRadians(0.25f * (y - mLastMousePos.y));

        // These angles are used at the beginning of Update() to set the view matrix.
        // It's just a pitch and yaw rotation of the camera boomed to the cube.
        mCameraPitch += xDiff;
        // Limit yaw rotation to 180 degrees
        mCameraYaw = MathHelper::Clamp(mCameraYaw + yDiff, 0.1f, MathHelper::Pi - 0.1f);
    }
    else if ((buttonState & MK_RBUTTON) != 0) {
        // Right click-and-drag = ZOOM
        float xDiff = 0.005f * (x - mLastMousePos.x);
        float yDiff = 0.005f * (y - mLastMousePos.y);

        // mRadius is distance of camera from the cube.
        // Clamp to min and max zoom. 2.5f "max" zoom allows for a little bit of clipping
        // on the corners of the box, and we like that.
        mCameraDistFromCube = MathHelper::Clamp(mCameraDistFromCube + (xDiff - yDiff), 2.5f, 23.f);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
}

void RainbowBoxDemo::UpdateCamera(const GameTimer& timer) {
    // mCameraYaw and mCameraPitch update via OnMouseMove; convert angles to Cartesian coords
    float x = mCameraDistFromCube * sinf(mCameraYaw) * cosf(mCameraPitch);
    float z = mCameraDistFromCube * sinf(mCameraYaw) * sinf(mCameraPitch);
    float y = mCameraDistFromCube * cosf(mCameraYaw);

    // Refresh view matrix based on translated mouse coords
    DirectX::XMVECTOR cameraPosition = DirectX::XMVectorSet(x, y, z, 1.f);
    DirectX::XMVECTOR lookAt = DirectX::XMVectorZero();
    DirectX::XMVECTOR upDir = DirectX::XMVectorSet(0.f, 1.f, 0.f, 0.f);

    DirectX::XMMATRIX V = DirectX::XMMatrixLookAtLH(cameraPosition, lookAt, upDir);
    XMStoreFloat4x4(&mView, V);
}

void RainbowBoxDemo::BuildDescriptorHeaps() {
    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = 1;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&cbvHeapDesc, IID_PPV_ARGS(&mCbvHeap)));
}

void RainbowBoxDemo::BuildConstantBuffers() {
    mObjectCB = std::make_unique<UploadBuffer<ObjectConstants>>(md3dDevice.Get(), 1, true);

    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));

    D3D12_GPU_VIRTUAL_ADDRESS cbAddress = mObjectCB->Resource()->GetGPUVirtualAddress();
    // Offset to the ith object constant buffer in the buffer.
    int boxCBufIndex = 0;
    cbAddress += boxCBufIndex * objCBByteSize;

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
    cbvDesc.BufferLocation = cbAddress;
    cbvDesc.SizeInBytes = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));

    md3dDevice->CreateConstantBufferView(
        &cbvDesc,
        mCbvHeap->GetCPUDescriptorHandleForHeapStart());
}

void RainbowBoxDemo::BuildRootSignature() {
    CD3DX12_ROOT_PARAMETER slotRootParameter[1];
    CD3DX12_DESCRIPTOR_RANGE cbvTable;
    cbvTable.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
    slotRootParameter[0].InitAsDescriptorTable(1, &cbvTable);
    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(1, slotRootParameter, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    Microsoft::WRL::ComPtr<ID3DBlob> serializedRootSig = nullptr;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

    if (errorBlob != nullptr) {
        ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());
    }
    ThrowIfFailed(hr);

    ThrowIfFailed(md3dDevice->CreateRootSignature(
        0,
        serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(),
        IID_PPV_ARGS(&mRootSignature)));
}

void RainbowBoxDemo::BuildShadersAndInputLayout() {
    HRESULT hr = S_OK;

    mvsByteCode = d3dUtil::CompileShader(L"src\\shader\\rainbowbox_color.hlsl", nullptr, "VS", "vs_5_0");
    mpsByteCode = d3dUtil::CompileShader(L"src\\shader\\rainbowbox_color.hlsl", nullptr, "PS", "ps_5_0");

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
    };
}

void RainbowBoxDemo::BuildBoxGeometry() {
    std::array<Vertex, 8> vertices =
    {
        Vertex({ DirectX::XMFLOAT3(-1.f, -1.f, -1.f), DirectX::XMFLOAT4(DirectX::Colors::White) }),
        Vertex({ DirectX::XMFLOAT3(-1.f, +1.f, -1.f), DirectX::XMFLOAT4(DirectX::Colors::Black) }),
        Vertex({ DirectX::XMFLOAT3(+1.f, +1.f, -1.f), DirectX::XMFLOAT4(DirectX::Colors::Red) }),
        Vertex({ DirectX::XMFLOAT3(+1.f, -1.f, -1.f), DirectX::XMFLOAT4(DirectX::Colors::Green) }),
        Vertex({ DirectX::XMFLOAT3(-1.f, -1.f, +1.f), DirectX::XMFLOAT4(DirectX::Colors::Blue) }),
        Vertex({ DirectX::XMFLOAT3(-1.f, +1.f, +1.f), DirectX::XMFLOAT4(DirectX::Colors::Yellow) }),
        Vertex({ DirectX::XMFLOAT3(+1.f, +1.f, +1.f), DirectX::XMFLOAT4(DirectX::Colors::Cyan) }),
        Vertex({ DirectX::XMFLOAT3(+1.f, -1.f, +1.f), DirectX::XMFLOAT4(DirectX::Colors::Magenta) })
    };

    std::array<std::uint16_t, 36> indices =
    {
        // front face
        0, 1, 2,
        0, 2, 3,

        // back face
        4, 6, 5,
        4, 7, 6,

        // left face
        4, 5, 1,
        4, 1, 0,

        // right face
        3, 2, 6,
        3, 6, 7,

        // top face
        1, 5, 6,
        1, 6, 2,

        // bottom face
        4, 0, 3,
        4, 3, 7
    };

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
    const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

    mBoxGeo = std::make_unique<MeshGeometry>();
    mBoxGeo->Name = "boxGeo";

    ThrowIfFailed(D3DCreateBlob(vbByteSize, &mBoxGeo->VertexBufferCPU));
    CopyMemory(mBoxGeo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

    ThrowIfFailed(D3DCreateBlob(ibByteSize, &mBoxGeo->IndexBufferCPU));
    CopyMemory(mBoxGeo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

    mBoxGeo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
        mCommandList.Get(), vertices.data(), vbByteSize, mBoxGeo->VertexBufferUploader);

    mBoxGeo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
        mCommandList.Get(), indices.data(), ibByteSize, mBoxGeo->IndexBufferUploader);

    mBoxGeo->VertexByteStride = sizeof(Vertex);
    mBoxGeo->VertexBufferByteSize = vbByteSize;
    mBoxGeo->IndexFormat = DXGI_FORMAT_R16_UINT;
    mBoxGeo->IndexBufferByteSize = ibByteSize;

    SubmeshGeometry submesh;
    submesh.IndexCount = (UINT)indices.size();
    submesh.StartIndexLocation = 0;
    submesh.BaseVertexLocation = 0;

    mBoxGeo->DrawArgs["box"] = submesh;
}

void RainbowBoxDemo::BuildPSO() {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC psoDesc;
    ZeroMemory(&psoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    psoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
    psoDesc.pRootSignature = mRootSignature.Get();
    psoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mvsByteCode->GetBufferPointer()),
        mvsByteCode->GetBufferSize()
    };
    psoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mpsByteCode->GetBufferPointer()),
        mpsByteCode->GetBufferSize()
    };
    psoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    psoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    psoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    psoDesc.SampleMask = UINT_MAX;
    psoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    psoDesc.NumRenderTargets = 1;
    psoDesc.RTVFormats[0] = mBackBufferFormat;
    psoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    psoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    psoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&psoDesc, IID_PPV_ARGS(&mPSO)));
}