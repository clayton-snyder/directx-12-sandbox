#include "d3dApp.h"
#include "MathHelper.h"
#include "UploadBuffer.h"
#include "GeometryGenerator.h"

#include "Waves.h"

const int gNumFrameResources = 3;




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

	// Same with the constant buffers and *dynamic* vertex buffer
	std::unique_ptr<UploadBuffer<PassConstants>> passCB = nullptr;
	std::unique_ptr<UploadBuffer<ObjectConstants>> objectCB = nullptr;
	std::unique_ptr<UploadBuffer<Vertex>> wavesVB = nullptr; 

	UINT64 fence = 0;

	FrameResource(ID3D12Device* device, UINT passCount, UINT objectCount, UINT waveVertCount) {
		ThrowIfFailed(device->CreateCommandAllocator(
			D3D12_COMMAND_LIST_TYPE_DIRECT,
			IID_PPV_ARGS(cmdListAlloc.GetAddressOf())));

		passCB = std::make_unique<UploadBuffer<PassConstants>>(device, passCount, true);
		objectCB = std::make_unique<UploadBuffer<ObjectConstants>>(device, objectCount, true);
		wavesVB = std::make_unique<UploadBuffer<Vertex>>(device, waveVertCount, false);
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







enum class RenderLayer : int {
	Opaque = 0,
	Count
};

class RiverDemo : public D3DApp {
public:
	RiverDemo(HINSTANCE hInstance);
	RiverDemo(const RiverDemo& rValue) = delete;
	RiverDemo& operator=(const RiverDemo& rValue) = delete;
	~RiverDemo();

	virtual bool Initialize()override;

private:
	virtual void OnResize()override;
	virtual void Update(const GameTimer& timer) override;
	virtual void Draw(const GameTimer& timer) override;

	virtual void OnMouseDown(WPARAM buttonState, int x, int y)override;
	virtual void OnMouseUp(WPARAM buttonState, int x, int y)override;
	virtual void OnMouseMove(WPARAM buttonState, int x, int y)override;

	void OnKeyboardInput(const GameTimer& timer);
	void UpdateCamera(const GameTimer& timer);
	void UpdateObjectCBs(const GameTimer& timer);
	void UpdateMainPassCB(const GameTimer& timer);
	void UpdateWaves(const GameTimer& timer);

	void BuildRootSignature();
	void BuildShadersAndInputLayout();
	void BuildLandGeometry();
	void BuildWavesGeometryBuffers();
	void BuildPSOs();
	void BuildFrameResources();
	void BuildRenderItems();
	void DrawRenderItems(ID3D12GraphicsCommandList* cmdList, const std::vector<RenderItem*>& ritems);

	float GetHillsHeight(float x, float z)const;
	DirectX::XMFLOAT3 GetHillsNormal(float x, float z)const;

private:

	std::vector<std::unique_ptr<FrameResource>> mFrameResources;
	FrameResource* mCurrFrameResource = nullptr;
	int mCurrFrameResourceIndex = 0;

	UINT mCbvSrvDescriptorSize = 0;

	Microsoft::WRL::ComPtr<ID3D12RootSignature> mRootSignature = nullptr;

	std::unordered_map<std::string, std::unique_ptr<MeshGeometry>> mGeometries;
	std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3DBlob>> mShaders;
	std::unordered_map<std::string, Microsoft::WRL::ComPtr<ID3D12PipelineState>> mPSOs;

	std::vector<D3D12_INPUT_ELEMENT_DESC> mInputLayout;

	RenderItem* mWavesRitem = nullptr;

	// List of all the render items.
	std::vector<std::unique_ptr<RenderItem>> mRenderItems;

	// Render items divided by PSO.
	std::vector<RenderItem*> mRitemLayer[(int)RenderLayer::Count];

	std::unique_ptr<Waves> mWaves;

	PassConstants mMainPassCB;

	bool mIsWireframe = false;

	DirectX::XMFLOAT3 mEyePos = { 0.f, 0.f, 0.f };
	DirectX::XMFLOAT4X4 mView = MathHelper::Identity4x4();
	DirectX::XMFLOAT4X4 mProj = MathHelper::Identity4x4();

	// Actually Pitch and Yaw aren't perfect terms here, but close enough.
	float mCameraPitch = 1.5f * DirectX::XM_PI;
	float mCameraYaw = 0.2 * DirectX::XM_PI;
	float mCameraDistFromCenter = 50.f;

	float mSunPitch = 1.25f * DirectX::XM_PI;
	float mSunYaw = DirectX::XM_PIDIV4;

	POINT mLastMousePos;
};

///* Uncomment to run this file as the main.Only one WinMain can exist in the project.
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance,
	PSTR cmdLine, int showCmd)
{
// Runtime memory check when debug
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	try {
		RiverDemo app(hInstance);
		if (!app.Initialize()) return 0;
		return app.Run();
	}
	catch (DxException& e) {
		MessageBox(nullptr, e.ToString().c_str(), L"Caught failed HResult!", MB_OK);
		return 0;
	}
}
//*/

RiverDemo::RiverDemo(HINSTANCE hInstance) : D3DApp(hInstance) {}

RiverDemo::~RiverDemo() {
	if (md3dDevice != nullptr) FlushCommandQueue();
}

bool RiverDemo::Initialize() {
	if (!D3DApp::Initialize()) return false;

	// Start with a fresh cmd list
	ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

	mWaves = std::make_unique<Waves>(128, 128, 1.f, 0.03f, 4.f, 0.2f);

	// todo??: does order matter here?
	BuildRootSignature();
	BuildShadersAndInputLayout();
	BuildLandGeometry();
	BuildWavesGeometryBuffers();
	BuildRenderItems();
	BuildRenderItems();
	BuildFrameResources();
	BuildPSOs();

	// Add initialization commands to the queue
	ThrowIfFailed(mCommandList->Close());
	ID3D12CommandList* cmdsLists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(cmdsLists), cmdsLists);

	// wait for init to finish.
	FlushCommandQueue();

	return true;
}

void RiverDemo::OnResize()
{
	D3DApp::OnResize();

	// The window resized, so update the aspect ratio and recompute the projection matrix.
	DirectX::XMMATRIX P = DirectX::XMMatrixPerspectiveFovLH(
		0.25f * MathHelper::Pi, AspectRatio(), 1.f, 1000.f);
	XMStoreFloat4x4(&mProj, P);
}

void RiverDemo::Update(const GameTimer& timer)
{
	OnKeyboardInput(timer);
	UpdateCamera(timer); // This updates mView

	// Advance to the next frame in the circular array (i.e., oldest remaining frame)
	mCurrFrameResourceIndex = (mCurrFrameResourceIndex + 1) % gNumFrameResources;
	mCurrFrameResource = mFrameResources[mCurrFrameResourceIndex].get();

	// Wait for GPU to catch up if it isn't already. Circular resource array should prevent
	// this from happening very much
	if (mCurrFrameResource->fence != 0 
		&& mFence->GetCompletedValue() < mCurrFrameResource->fence)
	{
		HANDLE eventHandle = CreateEventEx(nullptr, false, false, EVENT_ALL_ACCESS);
		ThrowIfFailed(mFence->SetEventOnCompletion(mCurrFrameResource->fence, eventHandle));
		WaitForSingleObject(eventHandle, INFINITE);
		CloseHandle(eventHandle);
	}

	UpdateObjectCBs(timer);
	UpdateMainPassCB(timer);
	UpdateWaves(timer);
}

void RiverDemo::Draw(const GameTimer& timer) {
	Microsoft::WRL::ComPtr<ID3D12CommandAllocator> cmdListAlloc = mCurrFrameResource->cmdListAlloc;

	// We can reset here because Update() is called before Draw(), and we ensure that the 
	// current FrameResource is done executing on the GPU before moving on. This is the 
	// proverbial "better way" mentioned in recent demos (i.e., circular resource array).
	ThrowIfFailed(cmdListAlloc->Reset());

	// Re-use command list memory; allocator was already reset so this is safe
	if (mIsWireframe) {
		ThrowIfFailed(mCommandList->Reset(cmdListAlloc.Get(), mPSOs["opaque_wireframe"].Get()));
	}
	else {
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
		DirectX::Colors::Aquamarine, 0, nullptr);
	mCommandList->ClearDepthStencilView(DepthStencilView(),
		D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.f, 0, 0, nullptr);

	// Set the pipeline: bind the back buffer as render target and use main DSV buffer 
	D3D12_CPU_DESCRIPTOR_HANDLE hBackBufferView = CurrentBackBufferView();
	D3D12_CPU_DESCRIPTOR_HANDLE hDepthStencilView = DepthStencilView();
	mCommandList->OMSetRenderTargets(1, &hBackBufferView, true, &hDepthStencilView);

	mCommandList->SetGraphicsRootSignature(mRootSignature.Get());

	auto passCB = mCurrFrameResource->passCB->Resource();
	mCommandList->SetGraphicsRootConstantBufferView(1, passCB->GetGPUVirtualAddress());

	DrawRenderItems(mCommandList.Get(), mRitemLayer[(int)RenderLayer::Opaque]);

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

void RiverDemo::OnMouseDown(WPARAM btnState, int x, int y) {
	mLastMousePos.x = x;
	mLastMousePos.y = y;
	SetCapture(mhMainWnd);
}

void RiverDemo::OnMouseUp(WPARAM btnState, int x, int y)
{
	ReleaseCapture();
}

void RiverDemo::OnMouseMove(WPARAM buttonState, int x, int y) {
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

void RiverDemo::OnKeyboardInput(const GameTimer& timer)
{
	// GetAsyncKeyState returns a short with most significant bit set if the key is down
	// Bitwise AND with 0x8000 to mask off everything but MSB. If non-zero, will return true.
	mIsWireframe = (GetAsyncKeyState('1') & 0x8000);
}

void RiverDemo::UpdateCamera(const GameTimer& timer)
{
	// Convert Spherical to Cartesian coordinates.
	mEyePos.x = mCameraDistFromCenter * sinf(mCameraYaw) * cosf(mCameraPitch);
	mEyePos.z = mCameraDistFromCenter * sinf(mCameraYaw) * sinf(mCameraPitch);
	mEyePos.y = mCameraDistFromCenter * cosf(mCameraYaw);

	// Build the view matrix.
	DirectX::XMVECTOR pos = DirectX::XMVectorSet(mEyePos.x, mEyePos.y, mEyePos.z, 1.f);
	DirectX::XMVECTOR target = DirectX::XMVectorZero();
	DirectX::XMVECTOR up = DirectX::XMVectorSet(0.f, 1.f, 0.f, 0.f);

	DirectX::XMMATRIX view = DirectX::XMMatrixLookAtLH(pos, target, up);
	XMStoreFloat4x4(&mView, view);
}

void RiverDemo::UpdateObjectCBs(const GameTimer& timer) {
	auto currObjectCB = mCurrFrameResource->objectCB.get();
	for (auto& renderItem :  mRenderItems) {
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
void RiverDemo::UpdateMainPassCB(const GameTimer& timer) {
	DirectX::XMMATRIX V = XMLoadFloat4x4(&mView);
	DirectX::XMMATRIX P = XMLoadFloat4x4(&mProj);

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

void RiverDemo::UpdateWaves(const GameTimer& timer) {
	// Add a new random wave every 0.25s
	static float t_base = 0.f;
	if ((mTimer.TotalTime() - t_base) >= 0.25f) {
		t_base += 0.25f;

		int i = MathHelper::Rand(4, mWaves->RowCount() - 5);
		int j = MathHelper::Rand(4, mWaves->ColumnCount() - 5);

		float r = MathHelper::RandF(0.2f, 0.5f);

		mWaves->Disturb(i, j, r);
	}

	mWaves->Update(timer.DeltaTime());

	UploadBuffer<Vertex>* currWavesVB = mCurrFrameResource->wavesVB.get();
	for (int i = 0; i < mWaves->VertexCount(); ++i) {
		Vertex v;

		v.position = mWaves->Position(i);
		v.color = DirectX::XMFLOAT4(DirectX::Colors::Blue);

		currWavesVB->CopyData(i, v);
	}

	mWavesRitem->geo->VertexBufferGPU = currWavesVB->Resource();
}

void RiverDemo::BuildRootSignature()
{
	// "Root parameter can be a table, root descriptor or root constants"
	CD3DX12_ROOT_PARAMETER slotRootParameter[2];
	slotRootParameter[0].InitAsConstantBufferView(0);
	slotRootParameter[1].InitAsConstantBufferView(1);


	CD3DX12_ROOT_SIGNATURE_DESC rootSigDesc(2, slotRootParameter, 0, nullptr, 
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

void RiverDemo::BuildShadersAndInputLayout() {
	mShaders["standardVS"] = d3dUtil::CompileShader(L"src\\shader\\river_color.hlsl", nullptr, "VS", "vs_5_0");
	mShaders["opaquePS"] = d3dUtil::CompileShader(L"src\\shader\\river_color.hlsl", nullptr, "PS", "ps_5_0");

	mInputLayout =
	{
		{ "POSITION", 0, DXGI_FORMAT_R32G32B32_FLOAT, 0, 0, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 },
		{ "COLOR", 0, DXGI_FORMAT_R32G32B32A32_FLOAT, 0, 12, D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA, 0 }
	};
}

void RiverDemo::BuildLandGeometry() {
	GeometryGenerator geoGen;
	GeometryGenerator::MeshData grid = geoGen.CreateGrid(160.0f, 160.0f, 50, 50);

	// Apply height function to each vertex anx color vertices accordingly (snowy peaks)
	std::vector<Vertex> vertices(grid.Vertices.size());
	for (size_t i = 0; i < grid.Vertices.size(); ++i) {
		auto& p = grid.Vertices[i].Position;
		vertices[i].position = p;
		vertices[i].position.y = GetHillsHeight(p.x, p.z);

		// Set vertex color based on height
		if (vertices[i].position.y < -10.f) {
			vertices[i].color = DirectX::XMFLOAT4(1.f, 0.96f, 0.62f, 1.f); // beach
		}
		else if (vertices[i].position.y < 5.f) {
			vertices[i].color = DirectX::XMFLOAT4(0.48f, 0.77f, 0.46f, 1.f); // light grass
		}
		else if (vertices[i].position.y < 12.f) {
			vertices[i].color = DirectX::XMFLOAT4(0.1f, 0.48f, 0.19f, 1.f); // dark grass
		}
		else if (vertices[i].position.y < 20.f) {
			vertices[i].color = DirectX::XMFLOAT4(0.45f, 0.39f, 0.34f, 1.f); // brown
		}
		else {
			vertices[i].color = DirectX::XMFLOAT4(1.f, 1.f, 1.f, 1.f); // snow
		}
	}

	const UINT vbByteSize = (UINT)vertices.size() * sizeof(Vertex);

	std::vector<std::uint16_t> indices = grid.GetIndices16();
	const UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

	auto geo = std::make_unique<MeshGeometry>();
	geo->Name = "landGeo";

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

	SubmeshGeometry submesh;
	submesh.IndexCount = (UINT)indices.size();
	submesh.StartIndexLocation = 0;
	submesh.BaseVertexLocation = 0;

	geo->DrawArgs["grid"] = submesh;

	mGeometries["landGeo"] = std::move(geo);
}

void RiverDemo::BuildWavesGeometryBuffers() {
	std::vector<std::uint16_t> indices(3 * mWaves->TriangleCount()); // 3 indices per face
	assert(mWaves->VertexCount() < 0x0000ffff);

	int m = mWaves->RowCount();
	int n = mWaves->ColumnCount();
	int k = 0;
	// For each quad...
	for (int i = 0; i < m - 1; ++i) {
		for (int j = 0; j < n - 1; ++j) {
			indices[k] = i * n + j;
			indices[k + 1] = i * n + j + 1;
			indices[k + 2] = (i + 1) * n + j;

			indices[k + 3] = (i + 1) * n + j;
			indices[k + 4] = i * n + j + 1;
			indices[k + 5] = (i + 1) * n + j + 1;

			k += 6; // skip forward one quad offset
		}
	}

	UINT vbByteSize = mWaves->VertexCount() * sizeof(Vertex);
	UINT ibByteSize = (UINT)indices.size() * sizeof(std::uint16_t);

	std::unique_ptr<MeshGeometry> geo = std::make_unique<MeshGeometry>();
	geo->Name = "waterGeo";
	geo->VertexBufferCPU = nullptr;
	geo->VertexBufferGPU = nullptr;

	ThrowIfFailed(D3DCreateBlob(ibByteSize, &geo->IndexBufferCPU));
	CopyMemory(geo->IndexBufferCPU->GetBufferPointer(), indices.data(), ibByteSize);

	geo->IndexBufferGPU = d3dUtil::CreateDefaultBuffer(md3dDevice.Get(),
		mCommandList.Get(), indices.data(), ibByteSize, geo->IndexBufferUploader);

	geo->VertexByteStride = sizeof(Vertex);
	geo->VertexBufferByteSize = vbByteSize;
	geo->IndexFormat = DXGI_FORMAT_R16_UINT;
	geo->IndexBufferByteSize = ibByteSize;

	SubmeshGeometry submesh;
	submesh.IndexCount = (UINT)indices.size();
	submesh.StartIndexLocation = 0;
	submesh.BaseVertexLocation = 0;

	geo->DrawArgs["grid"] = submesh;

	mGeometries["waterGeo"] = std::move(geo);
}

void RiverDemo::BuildPSOs() {
	D3D12_GRAPHICS_PIPELINE_STATE_DESC opaquePsoDesc;
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

	// for wireframes
	D3D12_GRAPHICS_PIPELINE_STATE_DESC opaqueWireframePsoDesc = opaquePsoDesc;
	opaqueWireframePsoDesc.RasterizerState.FillMode = D3D12_FILL_MODE_WIREFRAME;
	ThrowIfFailed(md3dDevice->CreateGraphicsPipelineState(&opaqueWireframePsoDesc, IID_PPV_ARGS(&mPSOs["opaque_wireframe"])));
}

void RiverDemo::BuildFrameResources() {
	for (int i = 0; i < gNumFrameResources; ++i) {
		mFrameResources.push_back(std::make_unique<FrameResource>(md3dDevice.Get(),
			1, (UINT)mRenderItems.size(), mWaves->VertexCount()));
	}
}

void RiverDemo::BuildRenderItems() {
	auto wavesRitem = std::make_unique<RenderItem>();
	wavesRitem->W = MathHelper::Identity4x4();
	wavesRitem->objCBIndex = 0;
	wavesRitem->geo = mGeometries["waterGeo"].get();
	wavesRitem->primitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	wavesRitem->indexCount = wavesRitem->geo->DrawArgs["grid"].IndexCount;
	wavesRitem->startIndexLocation = wavesRitem->geo->DrawArgs["grid"].StartIndexLocation;
	wavesRitem->baseVertexLocation = wavesRitem->geo->DrawArgs["grid"].BaseVertexLocation;

	mWavesRitem = wavesRitem.get();

	mRitemLayer[(int)RenderLayer::Opaque].push_back(wavesRitem.get());

	auto gridRitem = std::make_unique<RenderItem>();
	gridRitem->W = MathHelper::Identity4x4();
	gridRitem->objCBIndex = 1;
	gridRitem->geo = mGeometries["landGeo"].get();
	gridRitem->primitiveType = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
	gridRitem->indexCount = gridRitem->geo->DrawArgs["grid"].IndexCount;
	gridRitem->startIndexLocation = gridRitem->geo->DrawArgs["grid"].StartIndexLocation;
	gridRitem->baseVertexLocation = gridRitem->geo->DrawArgs["grid"].BaseVertexLocation;

	mRitemLayer[(int)RenderLayer::Opaque].push_back(gridRitem.get());

	mRenderItems.push_back(std::move(wavesRitem));
	mRenderItems.push_back(std::move(gridRitem));
}

void RiverDemo::DrawRenderItems(
	ID3D12GraphicsCommandList* cmdList, 
	const std::vector<RenderItem*>& renderItems)
{
	UINT objCBByteSize = d3dUtil::CalcConstantBufferByteSize(sizeof(ObjectConstants));

	auto objectCB = mCurrFrameResource->objectCB->Resource();

	// For each render item...
	for (size_t i = 0; i < renderItems.size(); ++i) {
		RenderItem* renderItem = renderItems[i];

		cmdList->IASetVertexBuffers(0, 1, &renderItem->geo->VertexBufferView());
		cmdList->IASetIndexBuffer(&renderItem->geo->IndexBufferView());
		cmdList->IASetPrimitiveTopology(renderItem->primitiveType);

		D3D12_GPU_VIRTUAL_ADDRESS objCBAddress = objectCB->GetGPUVirtualAddress();
		objCBAddress += renderItem->objCBIndex * objCBByteSize;

		cmdList->SetGraphicsRootConstantBufferView(0, objCBAddress);

		cmdList->DrawIndexedInstanced(
			renderItem->indexCount, 
			1, 
			renderItem->startIndexLocation, 
			renderItem->baseVertexLocation, 0);
	}
}

float RiverDemo::GetHillsHeight(float x, float z) const {
	return 0.3f * (z * sinf(0.1f * x) + x * cosf(0.1f * z));
}

DirectX::XMFLOAT3 RiverDemo::GetHillsNormal(float x, float z) const {
	DirectX::XMFLOAT3 n(
		-0.03f * z * cosf(0.1f * x) - 0.3f * cosf(0.1f * z),
		1.f,
		-0.3f * sinf(0.1f * x) + 0.03f * x * sinf(0.1f * z));

	DirectX::XMVECTOR unitNormal = DirectX::XMVector3Normalize(XMLoadFloat3(&n));
	DirectX::XMStoreFloat3(&n, unitNormal);

	return n;
}
