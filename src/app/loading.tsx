export default function Loading() {
  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-4 animate-pulse">
      <div className="h-12 w-12 rounded-full border-t-2 border-b-2 border-blue-500 animate-spin"></div>
      <p className="text-gray-500 font-medium">Initializing ML Visualizer...</p>
    </div>
  );
}
