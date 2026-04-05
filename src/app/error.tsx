'use client'

import { useEffect } from 'react'

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error(error)
  }, [error])

  return (
    <div className="flex flex-col items-center justify-center min-h-[60vh] space-y-6 text-center px-4">
      <h2 className="text-3xl font-bold text-gray-800">Something went wrong!</h2>
      <p className="text-gray-500 max-w-md">
        An error occurred while rendering the ML visualization. Try refreshing or resetting the application state.
      </p>
      <button
        onClick={() => reset()}
        className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition font-medium shadow-md hover:shadow-lg active:transform active:scale-95"
      >
        Try again
      </button>
    </div>
  )
}
