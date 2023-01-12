#include "d3dApp.h"
#include "MathHelper.h"
#include "UploadBuffer.h"
#include "GeometryGenerator.h"

// Uncomment this line only in the app you're running
//const int gNumFrameResources = 3;





///////////////////////////////////////////////////////////
////////////////////// FrameResource //////////////////////
///////////////////////////////////////////////////////////

struct ObjectConstants {
    DirectX::XMFLOAT4X4 world = MathHelper::Identity4x4();
};

// Constant data that is fixed over a given rendering pass
struct PassConstants {
    DirectX::XMFLOAT4X4 V = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 iV = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 P = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 iP = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 VP = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 iVP = MathHelper::Identity4x4();
    DirectX::XMFLOAT3 eyePosW = { 0.0f, 0.0f, 0.0f };
    float cbPerObjectPad1 = 0.0f;
    DirectX::XMFLOAT2 renderTargetSize = { 0.0f, 0.0f };
    DirectX::XMFLOAT2 invRenderTargetSize = { 0.0f, 0.0f };
    float nearZ = 0.0f;
    float farZ = 0.0f;
    float totalTime = 0.0f;
    float deltaTime = 0.0f;
};

struct Vertex {
    DirectX::XMFLOAT3 position;
    DirectX::XMFLOAT4 color;
};

// IMPORTANT CONCEPT: This stores all of the resources for a frame that can't be cleared until
// the GPU is done with it. We will keep several of these in a circular array so the CPU can 
// work on the next frame while the GPU is processing previous frames.
struct FrameResource {
    // Each frame gets its own allocator so we don't have to flush the queue before each frame
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmdListAlloc;

    // Same with the constant buffers
    std::unique_ptr<UploadBuffer<PassConstants>> passCB = nullptr;
    std::unique_ptr<UploadBuffer<ObjectConstants>> objectCB = nullptr;

    UINT64 fence = 0;

    FrameResource(ID3D12Device* device, UINT passCount, UINT objectCount) {
        ThrowIfFailed(device->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_DIRECT,
            IID_PPV_ARGS(cmdListAlloc.GetAddressOf())));

        passCB = std::make_unique<UploadBuffer<PassConstants>>(device, passCount, true);
        objectCB = std::make_unique<UploadBuffer<ObjectConstants>>(device, objectCount, true);
    }
    FrameResource(const FrameResource& rValue) = delete;
    FrameResource& operator=(const FrameResource& rValue) = delete;
    ~FrameResource() {}
};

//////////////////////////////////////////////////////////////////////////////////////////////






///////////////////////////////////////////////////////////
////////////////////// RenderItem /////////////////////////
///////////////////////////////////////////////////////////

struct RenderItem {
    RenderItem() = default;

    DirectX::XMFLOAT4X4 W = MathHelper::Identity4x4();  // World matrix

    // When object data changes, we need to update each FrameResource; this tracks remaining
    // needing to be updated
    int numStaleFrames = gNumFrameResources;
    UINT objCBIndex = -1;

    MeshGeometry* geo = nullptr;
    D3D12_PRIMITIVE_TOPOLOGY primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;

    // DrawIndexedInstanced() params
    UINT indexCount = 0;
    UINT startIndexLocation = 0;
    int baseVertexLocation = 0;
};

//////////////////////////////////////////////////////////////////////////////////////////////




class ShapesDemo : public D3DApp {
public:
    ShapesDemo(HINSTANCE hInstance);
    ShapesDemo(const ShapesDemo& rValue) = delete;
    ShapesDemo& operator=(const ShapesDemo& rValue) = delete;
    ~ShapesDemo();

    virtual bool Initialize() override;

private:
    virtual void OnResize() override;
    virtual void Update(const GameTimer& timer) override;
    virtual void Draw(const GameTimer& timer) override;

    virtual void OnMouseDown(WPARAM buttonState, int x, int y) override;
    virtual void OnMouseUp(WPARAM buttonState, int x, int y) override;
    virtual void OnMouseMove(WPARAM buttonState, int x, int y) override;

    void OnKeyboardInput(const GameTimer& timer);
    void UpdateCamera(const GameTimer& timer);
    void UpdateObjectCBs(const GameTimer& timer);
    void UpdateMainPassCB(const GameTimer& timer);

    void BuildDescriptorHeaps();
    void BuildConstantBufferViews();
    void BuildRootSignature();
    void BuildShadersAndInputLayout();
    void BuildShapeGeometry();
    void BuildPSOs();
    void BuildFrameResources();
    void BuildRenderItems();
    void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& renderItems);

private:
    // These are the fields necessary for the circular array used to dodge per-frame flushing
    std::vector<std::unique_ptr<FrameResource>> mFrameResources;
    FrameResource* mCurrFrameResource = nullptr;
    int mCurrFrameResourceIndex = 0;

    Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature = nullptr;
    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mCbvHeap = nullptr;

    Microsoft::WRL::ComPtr<ID3D12DescriptorHeap> mSrvDescriptorHeap = nullptr;

    std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3DBlob>> mShaders;
    std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12PipelineState>> mPSOs;

    std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

    // List of all the render items.
    std::vector<std::unique_ptr<RenderItem>> mRenderItems;

    // We'll keep render items that need different PSOs in different lists
    std::vector<RenderItem*> mOpaqueRenderItems;
    std::vector<RenderItem*> mTransparentRenderItems;

    PassConstants mMainPassCB;

    UINT mPassCbvOffset = 0;

    bool mIsWireframe = false;

    DirectX::XMFLOAT3 mEyePos = { 0.f, 0.f, 0.f };
    DirectX::XMFLOAT4X4 mView = MathHelper::Identity4x4();
    DirectX::XMFLOAT4X4 mProjection = MathHelper::Identity4x4();

    // Actually Pitch and Yaw aren't perfect terms here, but close enough.
    float mCameraPitch = 1.5f * DirectX::XM_PI;
    float mCameraYaw = 0.2 * DirectX::XM_PI;
    float mCameraDistFromCenter = 35.f;

    POINT mLastMousePos;
};

/* Uncomment to run this file as the main.Only one WinMain can exist in the project.
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
    PSTR cmdLine, int showCmd)
{
// Runtime memory check when debug
#if defined(DEBUG) | defined(_DEBUG)
    _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

    try {
        ShapesDemo app(hInstance);
        if (!app.Initialize()) return 0;
        return app.Run();
    }
    catch (DxException& e) {
        MessageBox(nullptr, e.ToString().c_str(), L"Caught failed HResult!", MB_OK);
        return 0;
    }
}
//*/

ShapesDemo::ShapesDemo(HINSTANCE hInstance) : D3DApp(hInstance) {}

ShapesDemo::~ShapesDemo() {
    if (md3dDevice != nullptr) FlushCommandQueue();
}

bool ShapesDemo::Initialize() {
    if (!D3DApp::Initialize()) return false;

    // Start with a fresh cmd list
    ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

    // todo??: does order matter here?
    BuildRootSignature();
    BuildShadersAndInputLayout();
    BuildShapeGeometry();
    BuildRenderItems();
    BuildFrameResources();
    BuildDescriptorHeaps();
    BuildConstantBufferViews();
    BuildPSOs();

    // Add initialization commands to the queue
    ThrowIfFailed(mCommandList->Close());
    ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
    mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

    // wait for init to finish.
    FlushCommandQueue();

    return true;
}

void ShapesDemo::OnResize() {
    D3DApp::OnResize();

    // The window resized, so update the aspect ratio and recompute the projection matrix.
    DirectX::XMMATRIX P = DirectX::XMMatrixPerspectiveFovLH(
        0.25f * MathHelper::Pi, AspectRatio(), 1.f, 1000.f);
    XMStoreFloat4x4(&mProjection, P);
}

void ShapesDemo::Update(const GameTimer& timer) {
    OnKeyboardInput(timer);
    UpdateCamera(timer); // This updates mView

    // Advance to the next frame in the circular array (i.e., oldest remaining frame)
    mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
    mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

    // Wait for GPU to catch up if it isn't already. Circular resource array should prevent
    // this from happening very much
    if (mCurrFrameResource->fence != 0 && 
        mFence->GetCompletedValue() < mCurrFrameResource->fence) 
    {
        HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
        ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->fence, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
        CloseHandle(eventHandle);
    }

    UpdateObjectCBs(timer);
    UpdateMainPassCB(timer);
}

void ShapesDemo::Draw(const GameTimer& timer) {
    Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmdListAlloc = mCurrFrameResource->cmdListAlloc;

    // We can reset here because Update() is called before Draw(), and we ensure that the 
    // current FrameResource is done executing on the GPU before moving on. This is the 
    // proverbial "better way" mentioned in recent demos (i.e., circular resource array).
    ThrowIfFailed(cmdListAlloc->Reset());

    // Re-use command list memory; allocator was already reset so this is safe
    if (mIsWireframe) {
        ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque_wireframe"].Get()));
    } else {
        ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque"].Get()));
    }

    // Gotta set the viewport and scissor rects again since we reset the command list
    mCommandList->RSSetViewports(1, &mScreenViewport);
    mCommandList->RSSetScissorRects(1, &mScissorRect);

    // Our first command is to transition the current back buffer into "render target" state
    // Currently it's in a "present" state leftover from being front buffer (last frame 
    // changed it to back buffer from a Present() command)
    CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
        CurrentBackBuffer(),
        D3D12_RESOURCE_STATE_PRESENT,
        D3D12_RESOURCE_STATE_RENDER_TARGET
    );
    mCommandList->ResourceBarrier(1, &transition);


    // Clear the back and depth/stencil buffers
    mCommandList->ClearRenderTargetView(CurrentBackBufferView(),
        DirectX::Colors::Azure, 0, nullptr);
    mCommandList->ClearDepthStencilView(DepthStencilView(),
        D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.f, 0, 0, nullptr);

    // Set the pipeline: bind the back buffer as render target and use main DSV buffer 
    D3D12_CPU_DESCRIPTOR_HANDLE hBackBufferView = CurrentBackBufferView();
    D3D12_CPU_DESCRIPTOR_HANDLE hDepthStencilView = DepthStencilView();
    mCommandList->OMSetRenderTargets(1, &hBackBufferView, true, &hDepthStencilView);

    ID3D12DescriptorHeap* descriptorHeaps[] = { mCbvHeap.Get() };
    mCommandList->SetDescriptorHeaps(_countof(descriptorHeaps), descriptorHeaps);

    mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

    int passCbvIndex = mPassCbvOffset + mCurrFrameResourceIndex;
    auto passCbvHandle = CD3DX12_GPU_DESCRIPTOR_HANDLE(mCbvHeap->GetGPUDescriptorHandleForHeapStart());
    passCbvHandle.Offset(passCbvIndex, mCbvSrvUavDescriptorSize);
    mCommandList->SetGraphicsRootDescriptorTable(1, passCbvHandle);

    DrawRenderItems(mCommandList.Get(), mOpaqueRenderItems);

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

    // Now swap the buffers
    ThrowIfFailed(mSwapChain->Present(0, 0));
    mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

    // Increment the CPU fence value
    mCurrFrameResource->fence = ++mCurrentFence;

    // This adds an instruction on the cmd queue to increment GPU's fence value. Since it's in
    // the queue it won't be incremented until the GPU gets here
    mCommandQueue->Signal(mFence.Get(), mCurrentFence);
}

void ShapesDemo::OnMouseDown(WPARAM buttonState, int x, int y) {
    mLastMousePos.x = x;
    mLastMousePos.y = y;
    SetCapture(mhMainWnd);
}

void ShapesDemo::OnMouseUp(WPARAM buttonState, int x, int y) {
    ReleaseCapture();
}

void ShapesDemo::OnMouseMove(WPARAM buttonState, int x, int y) {
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
        float xDiff = 0.05f * (x - mLastMousePos.x);
        float yDiff = 0.05f * (y - mLastMousePos.y);

        // mRadius is distance of camera from the cube.
        // Clamp to min and max zoom. 2.5f "max" zoom allows for a little bit of clipping
        // on the corners of the box, and we like that.
        mCameraDistFromCenter = MathHelper::Clamp(mCameraDistFromCenter + (xDiff - yDiff), 2.5f, 230.f);
    }

    mLastMousePos.x = x;
    mLastMousePos.y = y;
    
}

void ShapesDemo::OnKeyboardInput(const GameTimer& timer) {
    // GetAsyncKeyState returns a short with most significant bit set if the key is down
    // Bitwise AND with 0x8000 to mask off everything but MSB. If non-zero, will return true.
    mIsWireframe = (GetAsyncKeyState('1') & 0x8000);
}

void ShapesDemo::UpdateCamera(const GameTimer& timer) {
    // mCameraYaw and mCameraPitch update via OnMouseMove; convert angles to Cartesian coords
    mEyePos.x = mCameraDistFromCenter * sinf(mCameraYaw) * cosf(mCameraPitch);
    mEyePos.z = mCameraDistFromCenter * sinf(mCameraYaw) * sinf(mCameraPitch);
    mEyePos.y = mCameraDistFromCenter * cosf(mCameraYaw);

    // Refresh view matrix based on translated mouse coords
    DirectX::XMVECTOR cameraPosition = DirectX::XMVectorSet(mEyePos.x, mEyePos.y, mEyePos.z, 1.f);
    DirectX::XMVECTOR lookAt = DirectX::XMVectorZero();
    DirectX::XMVECTOR upDir = DirectX::XMVectorSet(0.f, 1.f, 0.f, 0.f);

    DirectX::XMMATRIX V = DirectX::XMMatrixLookAtLH(cameraPosition, lookAt, upDir);
    XMStoreFloat4x4(&mView, V);
}

// Call once per frame to update per-object buffers
void ShapesDemo::UpdateObjectCBs(const GameTimer& timer) {
    auto currObjectCB = mCurrFrameResource->objectCB.get();
    for (auto& renderItem : mRenderItems) {
        if (renderItem->numStaleFrames > 0) {
            DirectX::XMMATRIX world = XMLoadFloat4x4(&renderItem->W);

            ObjectConstants objConstants;
            XMStoreFloat4x4(&objConstants.world, XMMatrixTranspose(world));
            currObjectCB->CopyData(renderItem->objCBIndex, objConstants);

            renderItem->numStaleFrames--;
        }
    }
}

// Call once per frame to update per-pass buffers
void ShapesDemo::UpdateMainPassCB(const GameTimer& timer) {
    DirectX::XMMATRIX V = XMLoadFloat4x4(&mView);
    DirectX::XMMATRIX P = XMLoadFloat4x4(&mProjection);

    DirectX::XMMATRIX VP = XMMatrixMultiply(V, P);
    DirectX::XMMATRIX iV = XMMatrixInverse(&XMMatrixDeterminant(V), V);
    DirectX::XMMATRIX iP = XMMatrixInverse(&XMMatrixDeterminant(P), P);
    DirectX::XMMATRIX iVP = XMMatrixInverse(&XMMatrixDeterminant(VP), VP);

    XMStoreFloat4x4(&mMainPassCB.V, XMMatrixTranspose(V));
    XMStoreFloat4x4(&mMainPassCB.P, XMMatrixTranspose(P));
    XMStoreFloat4x4(&mMainPassCB.VP, XMMatrixTranspose(VP));
    XMStoreFloat4x4(&mMainPassCB.iV, XMMatrixTranspose(iV));
    XMStoreFloat4x4(&mMainPassCB.iP, XMMatrixTranspose(iP));
    XMStoreFloat4x4(&mMainPassCB.iVP, XMMatrixTranspose(iVP));
    mMainPassCB.eyePosW = mEyePos;
    mMainPassCB.renderTargetSize = DirectX::XMFLOAT2((float)mClientWidth, (float)mClientHeight);
    mMainPassCB.invRenderTargetSize = DirectX::XMFLOAT2(1.f / mClientWidth, 1.f / mClientHeight);
    mMainPassCB.nearZ = 1.f;
    mMainPassCB.farZ = 1000.f;
    mMainPassCB.totalTime = timer.TotalTime();
    mMainPassCB.deltaTime = timer.DeltaTime();

    auto currPassCB = mCurrFrameResource->passCB.get();
    currPassCB->CopyData(0, mMainPassCB);
}

void ShapesDemo::BuildDescriptorHeaps() {
    UINT numOpaqueRenderItems = (UINT)mOpaqueRenderItems.size();
    UINT numDescriptors = (numOpaqueRenderItems + 1) * gNumFrameResources;

    // Pass CBVs are the last 3 descriptors
    mPassCbvOffset = numOpaqueRenderItems * gNumFrameResources;

    D3D12_DESCRIPTOR_HEAP_DESC cbvHeapDesc;
    cbvHeapDesc.NumDescriptors = numDescriptors;
    cbvHeapDesc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    cbvHeapDesc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    cbvHeapDesc.NodeMask = 0;
    ThrowIfFailed(md3dDevice->CreateDescriptorHeap(&cbvHeapDesc, IID_PPV_ARGS(&mCbvHeap)));
}

void ShapesDemo::BuildConstantBufferViews() {
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    UINT numOpaqueRenderItems = (UINT)mOpaqueRenderItems.size();

    // Create CBV desc per object per FrameResource
    for (int frameIndex = 0; frameIndex < gNumFrameResources; ++frameIndex) {
        ID3D12Resource* objectCB = mFrameResources[frameIndex]->objectCB->Resource();
        for (UINT i = 0; i < numOpaqueRenderItems; i++) {
            D3D12_GPU_VIRTUAL_ADDRESS cbAddress = objectCB->GetGPUVirtualAddress();

            cbAddress += i * objCBByteSize;
            int heapIndex = (frameIndex * numOpaqueRenderItems) + i;
            CD3DX12_CPU_DESCRIPTOR_HANDLE heapHandle =
                CD3DX12_CPU_DESCRIPTOR_HANDLE(mCbvHeap->GetCPUDescriptorHandleForHeapStart());
            heapHandle.Offset(heapIndex, mCbvSrvUavDescriptorSize);

            D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
            cbvDesc.BufferLocation = cbAddress;
            cbvDesc.SizeInBytes = objCBByteSize;

            md3dDevice->CreateConstantBufferView(&cbvDesc, heapHandle);
        }
    }

    UINT passCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(PassConstants));
    for (int frameIndex = 0; frameIndex < gNumFrameResources; frameIndex++) {
        ID3D12Resource* passCB = mFrameResources[frameIndex]->passCB->Resource();
        D3D12_GPU_VIRTUAL_ADDRESS cbAddress = passCB->GetGPUVirtualAddress();

        int heapIndex = mPassCbvOffset + frameIndex;
        CD3DX12_CPU_DESCRIPTOR_HANDLE heapHandle = 
            CD3DX12_CPU_DESCRIPTOR_HANDLE(mCbvHeap->GetCPUDescriptorHandleForHeapStart());
        heapHandle.Offset(heapIndex, mCbvSrvUavDescriptorSize);

        D3D12_CONSTANT_BUFFER_VIEW_DESC cbvDesc;
        cbvDesc.BufferLocation = cbAddress;
        cbvDesc.SizeInBytes = passCBByteSize;

        md3dDevice->CreateConstantBufferView(&cbvDesc, heapHandle);
    }
}

void ShapesDemo::BuildRootSignature() {
    CD3DX12_DESCRIPTOR_RANGE cbvTable0;
    CD3DX12_DESCRIPTOR_RANGE cbvTable1;

    cbvTable0.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 0);
    cbvTable1.Init(D3D12_DESCRIPTOR_RANGE_TYPE_CBV, 1, 1);

    CD3DX12_ROOT_PARAMETER slotRootParam[2];
    slotRootParam[0].InitAsDescriptorTable(1, &cbvTable0);
    slotRootParam[1].InitAsDescriptorTable(1, &cbvTable1);

    CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2, slotRootParam, 0, nullptr,
        D3D12_ROOT_SIGNATURE_FLAG_ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT);

    Microsoft::WRL::ComPtr<ID3DBlob> serializedRootSig = nullptr;
    Microsoft::WRL::ComPtr<ID3DBlob> errorBlob = nullptr;
    HRESULT hResult = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
        serializedRootSig.GetAddressOf(), errorBlob.GetAddressOf());

    if (errorBlob != nullptr) ::OutputDebugStringA((char*)errorBlob->GetBufferPointer());

    ThrowIfFailed(hResult);
    ThrowIfFailed(md3dDevice->CreateRootSignature(0, serializedRootSig->GetBufferPointer(),
        serializedRootSig->GetBufferSize(), IID_PPV_ARGS(mRootSignature.GetAddressOf())));
}

void ShapesDemo::BuildShadersAndInputLayout() {
    mShaders["standardVS"] = d3dUtil::CompileShader(L"src\\shader\\shapes_color.hlsl", nullptr, "VS", "vs_5_1");
    mShaders["opaquePS"] = d3dUtil::CompileShader(L"src\\shader\\shapes_color.hlsl", nullptr, "PS", "ps_5_1");

    mInputLayout =
    {
        { "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
        { "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
    };
}

void ShapesDemo::BuildShapeGeometry() {
    GeometryGenerator geoGen;
    GeometryGenerator::MeshData box = geoGen.CreateBox(1.5f, 0.5f, 1.5f, 3);
    GeometryGenerator::MeshData grid = geoGen.CreateGrid(20.f, 30.f, 60, 40);
    GeometryGenerator::MeshData sphere = geoGen.CreateSphere(0.5f, 20, 20);
    GeometryGenerator::MeshData cylinder = geoGen.CreateCylinder(0.5f, 0.3f, 3.f, 20, 20);

    //
    // We are concatenating all the geometry into one big vertex/index buffer.  So
    // define the regions in the buffer each submesh covers.
    //

    // Cache the vertex offsets to each object in the concatenated vertex buffer.
    UINT boxVertexOffset = 0;
    UINT gridVertexOffset = (UINT)box.Vertices.size();
    UINT sphereVertexOffset = gridVertexOffset + (UINT)grid.Vertices.size();
    UINT cylinderVertexOffset = sphereVertexOffset + (UINT)sphere.Vertices.size();

    // Cache the starting index for each object in the concatenated index buffer.
    UINT boxIndexOffset = 0;
    UINT gridIndexOffset = (UINT)box.Indices32.size();
    UINT sphereIndexOffset = gridIndexOffset + (UINT)grid.Indices32.size();
    UINT cylinderIndexOffset = sphereIndexOffset + (UINT)sphere.Indices32.size();

    // Define the SubmeshGeometry that cover different 
    // regions of the vertex/index buffers.

    SubmeshGeometry boxSubmesh;
    boxSubmesh.IndexCount = (UINT)box.Indices32.size();
    boxSubmesh.StartIndexLocation = boxIndexOffset;
    boxSubmesh.BaseVertexLocation = boxVertexOffset;

    SubmeshGeometry gridSubmesh;
    gridSubmesh.IndexCount = (UINT)grid.Indices32.size();
    gridSubmesh.StartIndexLocation = gridIndexOffset;
    gridSubmesh.BaseVertexLocation = gridVertexOffset;

    SubmeshGeometry sphereSubmesh;
    sphereSubmesh.IndexCount = (UINT)sphere.Indices32.size();
    sphereSubmesh.StartIndexLocation = sphereIndexOffset;
    sphereSubmesh.BaseVertexLocation = sphereVertexOffset;

    SubmeshGeometry cylinderSubmesh;
    cylinderSubmesh.IndexCount = (UINT)cylinder.Indices32.size();
    cylinderSubmesh.StartIndexLocation = cylinderIndexOffset;
    cylinderSubmesh.BaseVertexLocation = cylinderVertexOffset;

    //
    // Extract the vertex elements we are interested in and pack the
    // vertices of all the meshes into one vertex buffer.
    //

    auto totalVertexCount =
        box.Vertices.size() +
        grid.Vertices.size() +
        sphere.Vertices.size() +
        cylinder.Vertices.size();

    std::vector<Vertex> vertices(totalVertexCount);

    UINT k = 0;
    for (size_t i = 0; i < box.Vertices.size(); ++i, ++k) {
        vertices[k].position = box.Vertices[i].Position;
        vertices[k].color = DirectX::XMFLOAT4(DirectX::Colors::IndianRed);
    }

    for (size_t i = 0; i < grid.Vertices.size(); ++i, ++k) {
        vertices[k].position = grid.Vertices[i].Position;
        vertices[k].color = DirectX::XMFLOAT4(DirectX::Colors::ForestGreen);
    }

    for (size_t i = 0; i < sphere.Vertices.size(); ++i, ++k) {
        vertices[k].position = sphere.Vertices[i].Position;
        vertices[k].color = DirectX::XMFLOAT4(DirectX::Colors::IndianRed);
    }

    for (size_t i = 0; i < cylinder.Vertices.size(); ++i, ++k) {
        vertices[k].position = cylinder.Vertices[i].Position;
        vertices[k].color = DirectX::XMFLOAT4(DirectX::Colors::SteelBlue);
    }

    std::vector<std::uint16_t> indices;
    indices.insert(indices.end(), std::begin(box.GetIndices16()), std::end(box.GetIndices16()));
    indices.insert(indices.end(), std::begin(grid.GetIndices16()), std::end(grid.GetIndices16()));
    indices.insert(indices.end(), std::begin(sphere.GetIndices16()), std::end(sphere.GetIndices16()));
    indices.insert(indices.end(), std::begin(cylinder.GetIndices16()), std::end(cylinder.GetIndices16()));

    const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);
    const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

    auto geo = std::make_unique<MeshGeometry>();
    geo->Name = "shapeGeo";

    ThrowIfFailed(D3DCreateBlob(vbByteSize, &geo->VertexBufferCPU));
    CopyMemory(geo->VertexBufferCPU->GetBufferPointer(), vertices.data(), vbByteSize);

    ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
    CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

    geo->VertexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
        mCommandList.Get(), vertices.data(), vbByteSize, geo->VertexBufferUploader);

    geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
        mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

    geo->VertexByteStride = sizeof(Vertex);
    geo->VertexBufferByteSize = vbByteSize;
    geo->IndexFormat = DXGI_FORMAT_R16_UINT;
    geo->IndexBufferByteSize = ibByteSize;

    geo->DrawArgs["box"] = boxSubmesh;
    geo->DrawArgs["grid"] = gridSubmesh;
    geo->DrawArgs["sphere"] = sphereSubmesh;
    geo->DrawArgs["cylinder"] = cylinderSubmesh;

    mGeometries[geo->Name] = std::move(geo);
}

void ShapesDemo::BuildPSOs() {
    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;

    //
    // PSO for opaque objects.
    //
    ZeroMemory(&opaquePsoDesc, sizeof(D3D12_GRAPHICS_PIPELINE_STATE_DESC));
    opaquePsoDesc.InputLayout = { mInputLayout.data(), (UINT)mInputLayout.size() };
    opaquePsoDesc.pRootSignature = mRootSignature.Get();
    opaquePsoDesc.VS =
    {
        reinterpret_cast<BYTE*>(mShaders["standardVS"]->GetBufferPointer()),
        mShaders["standardVS"]->GetBufferSize()
    };
    opaquePsoDesc.PS =
    {
        reinterpret_cast<BYTE*>(mShaders["opaquePS"]->GetBufferPointer()),
        mShaders["opaquePS"]->GetBufferSize()
    };
    opaquePsoDesc.RasterizerState = CD3DX12_RASTERIZER_DESC(D3D12_DEFAULT);
    opaquePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    opaquePsoDesc.BlendState = CD3DX12_BLEND_DESC(D3D12_DEFAULT);
    opaquePsoDesc.DepthStencilState = CD3DX12_DEPTH_STENCIL_DESC(D3D12_DEFAULT);
    opaquePsoDesc.SampleMask = UINT_MAX;
    opaquePsoDesc.PrimitiveTopologyType = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    opaquePsoDesc.NumRenderTargets = 1;
    opaquePsoDesc.RTVFormats[0] = mBackBufferFormat;
    opaquePsoDesc.SampleDesc.Count = m4xMsaaState ? 4 : 1;
    opaquePsoDesc.SampleDesc.Quality = m4xMsaaState ? (m4xMsaaQuality - 1) : 0;
    opaquePsoDesc.DSVFormat = mDepthStencilFormat;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaquePsoDesc, IID_PPV_ARGS(&mPSOs["opaque"])));


    //
    // PSO for opaque wireframe objects.
    //

    D3D12_GRAPHICS_PIPELINE_STATE_DESC opaqueWireframePsoDesc = opaquePsoDesc;
    opaqueWireframePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
    ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaqueWireframePsoDesc, IID_PPV_ARGS(&mPSOs["opaque_wireframe"])));
}

void ShapesDemo::BuildFrameResources() {
    for (int i = 0; i < gNumFrameResources; ++i) {
        mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
            1, (UINT)mRenderItems.size()));
    }
}

void ShapesDemo::BuildRenderItems() {
    auto boxRitem = std::make_unique<RenderItem>();
    XMStoreFloat4x4(&boxRitem->W, 
        DirectX::XMMatrixScaling(2.f, 2.f, 2.f) * DirectX::XMMatrixTranslation(0.f, 0.5f, 0.f));
    boxRitem->objCBIndex = 0;
    boxRitem->geo = mGeometries["shapeGeo"].get();
    boxRitem->primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    boxRitem->indexCount = boxRitem->geo->DrawArgs["box"].IndexCount;
    boxRitem->startIndexLocation = boxRitem->geo->DrawArgs["box"].StartIndexLocation;
    boxRitem->baseVertexLocation = boxRitem->geo->DrawArgs["box"].BaseVertexLocation;
    mRenderItems.push_back(std::move(boxRitem));

    auto gridRitem = std::make_unique<RenderItem>();
    gridRitem->W = MathHelper::Identity4x4();
    gridRitem->objCBIndex = 1;
    gridRitem->geo = mGeometries["shapeGeo"].get();
    gridRitem->primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    gridRitem->indexCount = gridRitem->geo->DrawArgs["grid"].IndexCount;
    gridRitem->startIndexLocation = gridRitem->geo->DrawArgs["grid"].StartIndexLocation;
    gridRitem->baseVertexLocation = gridRitem->geo->DrawArgs["grid"].BaseVertexLocation;
    mRenderItems.push_back(std::move(gridRitem));

    UINT objCBIndex = 2;
    for (int i = 0; i < 5; ++i) {
        auto leftCylRitem = std::make_unique<RenderItem>();
        auto rightCylRitem = std::make_unique<RenderItem>();
        auto leftSphereRitem = std::make_unique<RenderItem>();
        auto rightSphereRitem = std::make_unique<RenderItem>();

        DirectX::XMMATRIX leftCylWorld = DirectX::XMMatrixTranslation(-5.f, 1.5f, -10.f + i * 5.f);
        DirectX::XMMATRIX rightCylWorld = DirectX::XMMatrixTranslation(+5.f, 1.5f, -10.f + i * 5.f);

        DirectX::XMMATRIX leftSphereWorld = DirectX::XMMatrixTranslation(-5.f, 3.5f, -10.f + i * 5.f);
        DirectX::XMMATRIX rightSphereWorld = DirectX::XMMatrixTranslation(+5.f, 3.5f, -10.f + i * 5.f);

        XMStoreFloat4x4(&leftCylRitem->W, rightCylWorld);
        leftCylRitem->objCBIndex = objCBIndex++;
        leftCylRitem->geo = mGeometries["shapeGeo"].get();
        leftCylRitem->primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        leftCylRitem->indexCount = leftCylRitem->geo->DrawArgs["cylinder"].IndexCount;
        leftCylRitem->startIndexLocation = leftCylRitem->geo->DrawArgs["cylinder"].StartIndexLocation;
        leftCylRitem->baseVertexLocation = leftCylRitem->geo->DrawArgs["cylinder"].BaseVertexLocation;

        XMStoreFloat4x4(&rightCylRitem->W, leftCylWorld);
        rightCylRitem->objCBIndex = objCBIndex++;
        rightCylRitem->geo = mGeometries["shapeGeo"].get();
        rightCylRitem->primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        rightCylRitem->indexCount = rightCylRitem->geo->DrawArgs["cylinder"].IndexCount;
        rightCylRitem->startIndexLocation = rightCylRitem->geo->DrawArgs["cylinder"].StartIndexLocation;
        rightCylRitem->baseVertexLocation = rightCylRitem->geo->DrawArgs["cylinder"].BaseVertexLocation;

        XMStoreFloat4x4(&leftSphereRitem->W, leftSphereWorld);
        leftSphereRitem->objCBIndex = objCBIndex++;
        leftSphereRitem->geo = mGeometries["shapeGeo"].get();
        leftSphereRitem->primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        leftSphereRitem->indexCount = leftSphereRitem->geo->DrawArgs["sphere"].IndexCount;
        leftSphereRitem->startIndexLocation = leftSphereRitem->geo->DrawArgs["sphere"].StartIndexLocation;
        leftSphereRitem->baseVertexLocation = leftSphereRitem->geo->DrawArgs["sphere"].BaseVertexLocation;

        XMStoreFloat4x4(&rightSphereRitem->W, rightSphereWorld);
        rightSphereRitem->objCBIndex = objCBIndex++;
        rightSphereRitem->geo = mGeometries["shapeGeo"].get();
        rightSphereRitem->primitiveType = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
        rightSphereRitem->indexCount = rightSphereRitem->geo->DrawArgs["sphere"].IndexCount;
        rightSphereRitem->startIndexLocation = rightSphereRitem->geo->DrawArgs["sphere"].StartIndexLocation;
        rightSphereRitem->baseVertexLocation = rightSphereRitem->geo->DrawArgs["sphere"].BaseVertexLocation;

        mRenderItems.push_back(std::move(leftCylRitem));
        mRenderItems.push_back(std::move(rightCylRitem));
        mRenderItems.push_back(std::move(leftSphereRitem));
        mRenderItems.push_back(std::move(rightSphereRitem));
    }

    // We only have opaque RenderItems in this demo
    for (std::unique_ptr<RenderItem>& renderItem : mRenderItems)
        mOpaqueRenderItems.push_back(renderItem.get());
}

void ShapesDemo::DrawRenderItems(
    ID3D12GraphicsCommandList* cmdList, 
    const std::vector<RenderItem*>& renderItems) 
{
    UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));
    ID3D12Resource* objectCB = mCurrFrameResource->objectCB->Resource();

    for (int i = 0; i < renderItems.size(); i++) {
        RenderItem* renderItem = renderItems[i];

        cmdList->IASetVertexBuffers(0, 1, &renderItem->geo->VertexBufferView());
        cmdList->IASetIndexBuffer(&renderItem->geo->IndexBufferView());
        cmdList->IASetPrimitiveTopology(renderItem->primitiveType);

        // This object and FrameResource's CBV offset
        UINT cbvIndex = mCurrFrameResourceIndex * (UINT)mOpaqueRenderItems.size() + renderItem->objCBIndex;
        CD3DX12_GPU_DESCRIPTOR_HANDLE cbvHandle = 
            CD3DX12_GPU_DESCRIPTOR_HANDLE(mCbvHeap->GetGPUDescriptorHandleForHeapStart());
        cbvHandle.Offset(cbvIndex, mCbvSrvUavDescriptorSize);

        cmdList->SetGraphicsRootDescriptorTable(0, cbvHandle);

        cmdList->DrawIndexedInstanced(
            renderItem->indexCount, 
            1, 
            renderItem->startIndexLocation, 
            renderItem->baseVertexLocation, 0
        );
    }
}
