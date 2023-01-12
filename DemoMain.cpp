#include "include/d3dApp.h"

#include <iostream>
#include <DirectXColors.h>

class DemoMain : public D3DApp
{
public:
	DemoMain(HINSTANCE hInstance);
	~DemoMain();

	virtual bool Initialize() override;

private:
	virtual void OnResize() override;
	virtual void Update(const GameTimer& timer) override;
	virtual void Draw(const GameTimer& timer) override;
};

/* Uncomment to run this file as the main. Only one WinMain can exist in the project.
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE prevInstance, PSTR cmdLine, int showCmd) {
	std::cout << "Ages ago, life was born in the primitive sea." << std::endl;

// Runtime memory check for debug builds
#if defined(DEBUG) | defined(_DEBUG)
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
	std::cout << "DEBUG ON" << std::endl;
#endif

	try {
		DemoMain app(hInstance);
		if (!app.Initialize()) return 0;
		return app.Run();
	}
	catch (DxException& e) {
		MessageBox(nullptr, e.ToString().c_str(), L"HR Failed", MB_OK);
		return 0;
	}
}
*/

DemoMain::DemoMain(HINSTANCE hInstance) : D3DApp(hInstance) {}

DemoMain::~DemoMain() {}

bool DemoMain::Initialize() {
	return D3DApp::Initialize();
}

void DemoMain::OnResize() {
	D3DApp::OnResize();
}

void DemoMain::Update(const GameTimer& timer) {}

void DemoMain::Draw(const GameTimer& timer) {
	// I guess this implies that the command queue was flushed before each draw call? Since
	// resetting the command list allocator is bad if it's still in GPU queue
	ThrowIfFailed(mDirectCmdListAlloc->Reset());

	// Re-use command list memory; allocator was already reset so this is safe
	ThrowIfFailed(mCommandList->Reset(mDirectCmdListAlloc.Get(), nullptr));

	// Our first command is to transition the current back buffer into "render target" state
	// Currently it's in a "present" state leftover from being front buffer (last frame 
	// changed it to back buffer from a Present() command)
	CD3DX12_RESOURCE_BARRIER transition = CD3DX12_RESOURCE_BARRIER::Transition(
		CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_PRESENT,
		D3D12_RESOURCE_STATE_RENDER_TARGET
	);
	mCommandList->ResourceBarrier(1, &transition);

	// Apparently this has to happen "any time the command list is reset". Why?
	mCommandList->RSSetViewports(1, &mScreenViewport);
	mCommandList->RSSetScissorRects(1, &mScissorRect);

	// Clear the back and depth/stencil buffers
	mCommandList->ClearRenderTargetView(CurrentBackBufferView(), 
		DirectX::Colors::AliceBlue, 0, nullptr);
	mCommandList->ClearDepthStencilView(DepthStencilView(), 
		D3D12_CLEAR_FLAG_DEPTH | D3D12_CLEAR_FLAG_STENCIL, 1.f, 0, 0, nullptr);

	// Set the pipeline: bind the back buffer as render target and use main DSV buffer 
	D3D12_CPU_DESCRIPTOR_HANDLE hBackBufferView = CurrentBackBufferView();
	D3D12_CPU_DESCRIPTOR_HANDLE hDepthStencilView = DepthStencilView();
	mCommandList->OMSetRenderTargets(1, &hBackBufferView, true, &hDepthStencilView);

	// Transition the back buffer back to "present" state in preperation to swap 
	transition = CD3DX12_RESOURCE_BARRIER::Transition(
		CurrentBackBuffer(),
		D3D12_RESOURCE_STATE_RENDER_TARGET,
		D3D12_RESOURCE_STATE_PRESENT
	);
	mCommandList->ResourceBarrier(1, &transition);

	ThrowIfFailed(mCommandList->Close()); // Done recording commands

	// Add the command list to the command queue for execution
	ID3D12CommandList* cmdLists[] = { mCommandList.Get() };
	mCommandQueue->ExecuteCommandLists(_countof(cmdLists), cmdLists);
	// Remember at this point they are just submitted to the queue, not already executed

	// Now swap the buffers **(why do this before we know the cmd queue is flushed??)**
	ThrowIfFailed(mSwapChain->Present(0, 0));
	mCurrBackBuffer = (mCurrBackBuffer + 1) % SwapChainBufferCount;

	FlushCommandQueue();
}